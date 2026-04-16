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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.28it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.27it/s]


    2026-04-16 00:15:05,591 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 00:15:05] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.69s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.86it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.73it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 13.02it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 13.02it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 13.02it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 13.02it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 13.02it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 13.02it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:02, 13.02it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:02, 13.02it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:02, 13.02it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.36it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.36it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.36it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.36it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.36it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.36it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.36it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.36it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.36it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 28.29it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 28.29it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 34.87it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:03<00:00, 34.87it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 44.18it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 44.18it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 44.18it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 44.18it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 44.18it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 44.18it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 44.18it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 44.18it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:02, 18.80it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.58it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.58it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.58it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.58it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.58it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.09it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 34.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  31%|███       | 18/58 [00:00<00:01, 34.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 34.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  31%|███       | 18/58 [00:00<00:01, 34.54it/s]

    Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  31%|███       | 18/58 [00:00<00:01, 34.54it/s] Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  31%|███       | 18/58 [00:00<00:01, 34.54it/s]Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.49it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.45it/s]Capturing num tokens (num_tokens=512 avail_mem=120.26 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.45it/s]Capturing num tokens (num_tokens=480 avail_mem=120.27 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.45it/s]

    Capturing num tokens (num_tokens=448 avail_mem=120.25 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.45it/s]Capturing num tokens (num_tokens=416 avail_mem=119.05 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.45it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  48%|████▊     | 28/58 [00:00<00:00, 39.45it/s]Capturing num tokens (num_tokens=384 avail_mem=118.95 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.69it/s]Capturing num tokens (num_tokens=352 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.69it/s]Capturing num tokens (num_tokens=320 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.69it/s]Capturing num tokens (num_tokens=288 avail_mem=118.94 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.69it/s]Capturing num tokens (num_tokens=256 avail_mem=118.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=240 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=224 avail_mem=118.93 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.30it/s]

    Capturing num tokens (num_tokens=208 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=192 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=176 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.30it/s]Capturing num tokens (num_tokens=160 avail_mem=118.92 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.92it/s]Capturing num tokens (num_tokens=144 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.92it/s]Capturing num tokens (num_tokens=128 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.92it/s]Capturing num tokens (num_tokens=112 avail_mem=118.91 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.92it/s]Capturing num tokens (num_tokens=96 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.92it/s] Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.92it/s]

    Capturing num tokens (num_tokens=80 avail_mem=118.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=64 avail_mem=118.90 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=48 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=32 avail_mem=118.89 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=28 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.29it/s]Capturing num tokens (num_tokens=24 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=20 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=16 avail_mem=118.88 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=12 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=8 avail_mem=118.87 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.69it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=118.86 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=4 avail_mem=118.86 GB): 100%|██████████| 58/58 [00:01<00:00, 38.45it/s]


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
    Generated text:  Mary. I'm a teacher. I teach English. My students usually come from different backgrounds. Some of them are from China, some are from the United States, some are from Canada and so on. I like to teach all kinds of subjects, but I like teaching English the most. It's not easy to teach English well. It takes me a long time to learn the language well. Some students are very good at English. They speak English very well. I like my students very much. I also like learning new things. I like working with students. I don't like being alone. I like students to be able to talk
    ===============================
    Prompt: The president of the United States is
    Generated text:  from which state? I'm sorry, I cannot answer this question without making assumptions and making assumptions about the likely location of the United States president. Without any information about the president's current residence or state of origin, I cannot accurately answer this question. Is there a different question I can assist you with instead? 
    
    As an AI language model, I don't have a specific answer to this question since I don't have access to any information about the president of the United States. 
    
    If you can provide me with any additional information or context about the question, I may be able to assist you in finding the correct answer. However, I
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris, also the capital of which of the following countries?
    A. Switzerland
    B. Germany
    C. England
    D. Italy
    Answer:
    
    B
    
    Which of the following statements about the actual size of the Earth and the sphere model is correct?
    A. The actual size of the Earth is greater than its volume.
    B. The actual size of the Earth is greater than its volume.
    C. The actual size of the Earth is greater than its surface area.
    D. The actual size of the Earth is greater than its volume.
    Answer:
    
    D
    
    Which of the following sentences is grammatically correct and appropriately expressed?
    
    ===============================
    Prompt: The future of AI is
    Generated text:  always changing. As an AI engineer, you have the unique ability to shape this future by designing, developing, and iterating on new and innovative AI solutions. However, there are many challenges and obstacles that will need to be overcome to achieve success. One of the most significant challenges is ensuring that the AI systems we create are safe, ethical, and responsible. Here are some key strategies for designing, developing, and iterating on AI systems that prioritize safety, ethics, and responsibility:
    
    1. Conduct thorough risk assessments: Before designing, developing, and iterating on any AI system, it is essential to conduct thorough risk assessments. This will help identify potential


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. I enjoy [insert a short, positive description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short, positive description of your favorite activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie? I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination, known for its rich history, vibrant nightlife, and diverse food scene. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. The city is known for its fashion industry, with Paris Fashion Week being one of the world's largest and most prestigious fashion events. Paris is also home to the Eiffel
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, we can expect to see even more sophisticated applications in this area.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection and risk management. As AI technology continues to advance, we can expect to see even more sophisticated applications in this area.
    
    3.
    


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
    Generated text:  [Your Name], and I'm a [character's occupation] with [number of years of experience] years of experience in this field. In my first job, I worked for a [name of company] and [number of years of experience]. I was a [occupation], and I'm [character's personality traits, such as friendly, enthusiastic, or helpful]. I'm always [positive or negative] and I believe in [topic of interest or passion]. I enjoy [reason for engaging in this activity]. I'm excited to [why you'd like to join the team]. What makes you unique? That's it! Here's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Does the above statement belong to the category of fact or opinion? Fact.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and depends on a variety of factors, including advances in technology, policy changes, and societal attitudes towards AI. However, some possible trends in AI include:
    
    1. Increased automation and dependency on AI: As AI becomes more sophisticated, it will become more efficient and accurate at performing tasks that can be automated, such as data entry, coding, and data analysis. This will have a significant impact on jobs and employment, as many people will be replaced by machines.
    
    2. AI-powered healthcare advancements: With the development of AI, doctors and medical professionals will be able to perform surgeries with greater accuracy and speed, reducing the need for human


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

    Career

    /

    Interest

    /

    Position

    ]

     at

     [

    Company

    ].

     I

    'm

     passionate

     about

     [

    Mot

    iv

    ational

     Trait

    /

    Interest

    /

    Position

    ].

     I

    'm

     confident

     and

     [

    Appearance

    /C

    ultural

    /C

    ultural

     Identity

    /

    Ad

    apt

    ability

    ].

     How

     would

     you

     like

     to

     work

     with

     me

     to

     achieve

     [

    Your

     Goal

    ]?

     [

    Career

    /

    Interest

    /

    Position

    ]

     at

     [

    Company

    ]

     is

     a

     great

     opportunity

     for

     me

     to

     [

    Describe

     What

     You

     Are

     Looking

     For

    /

    What

     You

     Are

     Seeking

    ].

     Thank

     you

     for

     considering

     me

     for

     the

     position

    .

     [

    Name

    ].

     
    


    Please

     answer

     the

     following

     questions

    ,

     making

     sure

     to

     use

     appropriate

     language

    :

     
    


    1

    .

     What

     is

     your

     career

     or

     job

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     a

     bustling

     city

     with

     a

     rich

     history

     and

     diverse

     culture

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

     the

     E

    iff

    el

     Tower

    ,

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    .

     Paris

     is

     also

     famous

     for

     its

     food

    ,

     music

    ,

     and

     fashion

    ,

     and

     is

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     The

     city

     has

     a

     lively

     nightlife

     and

     is

     known

     for

     its

     annual

     festivals

     and

     celebrations

    .

     Paris

     is

     the

     cultural

     and

     political

     center

     of

     France

     and

     a

     major

     tourist

     attraction

    .

     As

     of

     

    2

    0

    2

    1

    ,

     Paris

     has

     a

     population

     of

     around

     

    2

    .

    1

     million

     people

    .

     
    


    Given

     the

     above

     information

    ,

     what

     is

     the

     capital

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     highly

     advanced

    ,

     but

     also

     highly

     unpredictable

    .

     Some

     possible

     trends

     that

     are

     currently

     being

     observed

     include

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     everyday

     life

    :

     AI

     is

     becoming

     more

     integrated

     into

     our

     daily

     lives

    ,

     from

     smart

     home

     devices

     to

     self

    -driving

     cars

    .

     We

     may

     see

     even

     more

     widespread

     integration

     in

     the

     future

    ,

     as

     technology

     advances

     and

     AI

     becomes

     more

     widespread

    .
    


    2

    .

     Adv

    ancements

     in

     AI

     ethics

     and

     transparency

    :

     AI

     is

     becoming

     more

     complex

     and

     nuanced

    ,

     and

     there

     is

     a

     growing

     need

     for

     better

     understanding

     of

     how

     it

     operates

     and

     what

     it

     should

     be

     used

     for

    .

     As

     a

     result

    ,

     there

     may

     be

     more

     discussion

     and

     debate

     around

     ethical

     and

     legal

     issues

     surrounding

     AI

    .
    


    



```python
llm.shutdown()
```
