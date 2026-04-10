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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.49it/s]


    2026-04-10 20:17:33,068 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 20:17:33] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.55it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.55it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.55it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:03, 12.55it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 19.54it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 19.54it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 19.54it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 19.54it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 19.54it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 19.54it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 19.54it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 19.54it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:03<00:00, 26.09it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:03<00:00, 26.09it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:03<00:00, 26.09it/s]

    Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:03<00:00, 26.09it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:03<00:00, 26.09it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:03<00:00, 26.09it/s]Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:03<00:00, 26.09it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:03<00:00, 26.09it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 29.97it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 34.21it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 34.21it/s]

    Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 34.21it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 34.21it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 34.21it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 34.21it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 34.21it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 34.21it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:03<00:00, 40.48it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:03<00:00, 40.48it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:03<00:00, 40.48it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:03<00:00, 40.48it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:03<00:00, 40.48it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.71 GB):   3%|▎         | 2/58 [00:00<00:02, 19.37it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.71 GB):   3%|▎         | 2/58 [00:00<00:02, 19.37it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.71 GB):   3%|▎         | 2/58 [00:00<00:02, 19.37it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.71 GB):   3%|▎         | 2/58 [00:00<00:02, 19.37it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=55.71 GB):   9%|▊         | 5/58 [00:00<00:02, 18.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.70 GB):   9%|▊         | 5/58 [00:00<00:02, 18.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.71 GB):   9%|▊         | 5/58 [00:00<00:02, 18.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.70 GB):   9%|▊         | 5/58 [00:00<00:02, 18.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.70 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.70 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.70 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.69 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.69 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.70it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=55.69 GB):  14%|█▍        | 8/58 [00:00<00:02, 21.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.69 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.69 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.68 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.68 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.67 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.67 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.67 GB):  31%|███       | 18/58 [00:00<00:01, 35.64it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.67 GB):  31%|███       | 18/58 [00:00<00:01, 35.64it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.66 GB):  31%|███       | 18/58 [00:00<00:01, 35.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.64 GB):  31%|███       | 18/58 [00:00<00:01, 35.64it/s]Capturing num tokens (num_tokens=960 avail_mem=55.66 GB):  31%|███       | 18/58 [00:00<00:01, 35.64it/s] Capturing num tokens (num_tokens=896 avail_mem=55.66 GB):  31%|███       | 18/58 [00:00<00:01, 35.64it/s]

    Capturing num tokens (num_tokens=832 avail_mem=55.65 GB):  31%|███       | 18/58 [00:00<00:01, 35.64it/s]Capturing num tokens (num_tokens=832 avail_mem=55.65 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.23it/s]Capturing num tokens (num_tokens=768 avail_mem=55.65 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.23it/s]Capturing num tokens (num_tokens=704 avail_mem=55.65 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.23it/s]Capturing num tokens (num_tokens=640 avail_mem=55.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.23it/s]Capturing num tokens (num_tokens=576 avail_mem=55.62 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.23it/s]Capturing num tokens (num_tokens=512 avail_mem=55.61 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.23it/s]Capturing num tokens (num_tokens=512 avail_mem=55.61 GB):  50%|█████     | 29/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=480 avail_mem=55.62 GB):  50%|█████     | 29/58 [00:00<00:00, 38.42it/s]

    Capturing num tokens (num_tokens=448 avail_mem=55.62 GB):  50%|█████     | 29/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=416 avail_mem=55.62 GB):  50%|█████     | 29/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=384 avail_mem=55.62 GB):  50%|█████     | 29/58 [00:00<00:00, 38.42it/s]Capturing num tokens (num_tokens=384 avail_mem=55.62 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.16it/s]Capturing num tokens (num_tokens=352 avail_mem=55.61 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.16it/s]Capturing num tokens (num_tokens=320 avail_mem=55.61 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.16it/s]Capturing num tokens (num_tokens=288 avail_mem=55.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.16it/s]

    Capturing num tokens (num_tokens=256 avail_mem=55.60 GB):  57%|█████▋    | 33/58 [00:01<00:00, 35.16it/s]Capturing num tokens (num_tokens=256 avail_mem=55.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=240 avail_mem=55.60 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=224 avail_mem=55.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=208 avail_mem=55.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=192 avail_mem=55.59 GB):  64%|██████▍   | 37/58 [00:01<00:00, 32.28it/s]Capturing num tokens (num_tokens=192 avail_mem=55.59 GB):  71%|███████   | 41/58 [00:01<00:00, 31.38it/s]Capturing num tokens (num_tokens=176 avail_mem=55.59 GB):  71%|███████   | 41/58 [00:01<00:00, 31.38it/s]

    Capturing num tokens (num_tokens=160 avail_mem=55.58 GB):  71%|███████   | 41/58 [00:01<00:00, 31.38it/s]Capturing num tokens (num_tokens=144 avail_mem=55.58 GB):  71%|███████   | 41/58 [00:01<00:00, 31.38it/s]Capturing num tokens (num_tokens=128 avail_mem=55.58 GB):  71%|███████   | 41/58 [00:01<00:00, 31.38it/s]Capturing num tokens (num_tokens=128 avail_mem=55.58 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=112 avail_mem=55.58 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=96 avail_mem=55.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.40it/s] Capturing num tokens (num_tokens=80 avail_mem=55.57 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.40it/s]

    Capturing num tokens (num_tokens=64 avail_mem=55.56 GB):  78%|███████▊  | 45/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=64 avail_mem=55.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 25.91it/s]Capturing num tokens (num_tokens=48 avail_mem=55.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 25.91it/s]Capturing num tokens (num_tokens=32 avail_mem=55.56 GB):  84%|████████▍ | 49/58 [00:01<00:00, 25.91it/s]

    Capturing num tokens (num_tokens=28 avail_mem=55.55 GB):  84%|████████▍ | 49/58 [00:01<00:00, 25.91it/s]Capturing num tokens (num_tokens=28 avail_mem=55.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 17.97it/s]Capturing num tokens (num_tokens=24 avail_mem=55.52 GB):  90%|████████▉ | 52/58 [00:01<00:00, 17.97it/s]Capturing num tokens (num_tokens=20 avail_mem=55.52 GB):  90%|████████▉ | 52/58 [00:02<00:00, 17.97it/s]Capturing num tokens (num_tokens=16 avail_mem=55.52 GB):  90%|████████▉ | 52/58 [00:02<00:00, 17.97it/s]

    Capturing num tokens (num_tokens=16 avail_mem=55.52 GB):  95%|█████████▍| 55/58 [00:02<00:00, 17.29it/s]Capturing num tokens (num_tokens=12 avail_mem=55.51 GB):  95%|█████████▍| 55/58 [00:02<00:00, 17.29it/s]Capturing num tokens (num_tokens=8 avail_mem=55.51 GB):  95%|█████████▍| 55/58 [00:02<00:00, 17.29it/s] Capturing num tokens (num_tokens=4 avail_mem=55.51 GB):  95%|█████████▍| 55/58 [00:02<00:00, 17.29it/s]Capturing num tokens (num_tokens=4 avail_mem=55.51 GB): 100%|██████████| 58/58 [00:02<00:00, 18.72it/s]Capturing num tokens (num_tokens=4 avail_mem=55.51 GB): 100%|██████████| 58/58 [00:02<00:00, 25.39it/s]


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
    Generated text:  Tim and I am a second year student at the University of Illinois at Chicago studying the brain and behavior. This semester, I am taking a class on how the brain processes human memory and thought. I have done some reading on the topic and have been thinking about how I might use this knowledge to understand and explain the emotion you feel when you are thinking of something.
    I understand that emotions can be a difficult topic to explain because people may have different types of emotions and they may not always be able to explain their emotions. In addition, emotions are often socially constructed and can be interpreted in different ways by different people. In my class, we
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He is like the boss of the country. He has to make important decisions that will affect the country. He is supposed to work with other important people to make decisions. He is supposed to decide important things such as the government policy and the country's budget.
    How many people are supposed to work with the president to make decisions?
    
    The president of the United States is supposed to work with many important people to make decisions. This includes the heads of other government agencies, as well as important people from the federal and state governments, as well as other individuals and organizations that the president might consult or work with
    ===============================
    Prompt: The capital of France is
    Generated text:  ________.
    A. Paris
    B. London
    C. Rome
    D. Berlin
    Answer:
    A
    
    According to the 'Shandong Province Safety Production Regulation', in the event of a production safety accident, the production and operation unit shall ____.
    A. Immediately rescue people in danger
    B. Organize rescue efforts and report to the local people's government
    C. Immediately organize rescue efforts, take measures to reduce casualties and property losses, and promptly report the accident situation to the local people's government
    D. Immediately organize rescue efforts, take measures to reduce casualties and property losses, and promptly report the accident situation to the
    ===============================
    Prompt: The future of AI is
    Generated text:  not only about the rise of AI, but it is also about the impact that it will have on the human race. As AI becomes more advanced, it will become increasingly important for businesses to understand the ethical implications of their use. It is essential to have a clear understanding of the ethical considerations that arise when using AI, and to ensure that these considerations are taken into account when developing and implementing AI systems. In this article, we will explore some of the key ethical considerations that arise when using AI, and how businesses can ensure that their AI systems are ethical and responsible.
    One of the most significant ethical considerations that arise when using AI is privacy


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or experience here]. I enjoy [insert a short description of your hobbies or interests here]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always up for new experiences and adventures. What's your favorite book or movie? I love [insert a short description of your favorite book or movie here]. I'm always looking for new ways to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic center with a rich history dating back to the Roman Empire and the French Revolution. It is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its cuisine, fashion, and art, and is home to numerous museums, theaters, and other cultural institutions. Paris is a major transportation hub, with the Eiffel Tower serving as a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and experiences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater reliance on data: AI systems will become increasingly reliant on large amounts of data to learn and make decisions. This will require more advanced data processing and storage technologies, as well as more sophisticated machine learning algorithms.
    
    3. Increased ethical considerations: As AI systems become more integrated with human intelligence, there will be increased pressure to consider
    


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
    Generated text:  [Your Name], and I'm a [Career Goal] person who has always had an interest in [Field/Subject]. I'm currently [Current Position] and I'm eager to explore new experiences and broaden my horizons, both professionally and personally. What makes you unique in your field? I'm passionate about [What motivates you to pursue this goal?]. What is your ultimate goal for the future, and how are you working towards it? I'm always looking for opportunities to grow and learn, and I'm excited to see what kind of impact I can have on the world. How do you stay motivated to pursue your goals
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and the second-largest city in Europe. Paris is also the birthplace of many notable figures, including the French composer Louis Chaumet, the French actress Julie Besson, and the French politician Charles de Gaulle. It is also known as "La Petiteville" due to its quaint countryside surroundings. Paris is a major cultural and economic center of France, with many museums, parks, and landmarks. It is also known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. Paris is a popular tourist destination, and the city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising and has several possible trends:
    
    1. Increased integration with human intelligence: As AI continues to improve, it is expected that it will become more integrated with human intelligence, allowing it to better understand and interpret human emotions, language, and thoughts.
    
    2. Development of more advanced AI systems: AI systems are becoming more sophisticated and powerful, with the potential to solve complex problems that were previously beyond the scope of human capabilities. As such, we may see the development of more advanced AI systems that can tackle a wide range of challenges.
    
    3. Expansion of AI into more industries: AI is becoming more prevalent in various industries, including healthcare,


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

     Jane

    ,

     and

     I

    'm

     

    3

    5

     years

     old

    .

     I

    'm

     a

     marketing

     professional

     with

     a

     passion

     for

     technology

     and

     design

    .

     I

     like

     to

     use

     my

     skills

     to

     help

     businesses

     get

     their

     messages

     across

     effectively

     and

     create

     engaging

     digital

     products

    .

     I

    'm

     always

     looking

     for

     new

     ways

     to

     improve

     my

     skills

     and

     stay

     up

    -to

    -date

     with

     the

     latest

     trends

     in

     marketing

     and

     design

    .

     If

     you

     have

     any

     questions

     or

     need

     assistance

     with

     something

    ,

     feel

     free

     to

     reach

     out

     to

     me

    .

     I

     look

     forward

     to

     potentially

     working

     with

     you

    !

     Q

    :

     What

     are

     some

     of

     the

     ways

     Jane

     utilizes

     her

     skills

     to

     help

     businesses

     achieve

     their

     marketing

     and

     design

     goals

    ?

     Answer

     in

     a

     phrase

     or

     two

    .

     Jane

     uses

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     city

     of

     light

     and

     the

     city

     of

     light

    .

     It

     is

     a

     historic

     and

     cultural

     center

     with

     a

     rich

     history

     and

     numerous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     known

     for

     its

     op

    ulent

     dining

     and

     nightlife

    ,

     as

     well

     as

     its

     vibrant

     arts

     scene

    ,

     and

     is

     a

     major

     transportation

     hub

    .

     The

     city

     is

     a

     hub

     for

     business

    ,

     culture

    ,

     and

     entertainment

    ,

     and

     is

     home

     to

     many

     famous

     museums

    ,

     theaters

    ,

     and

     coffee

     shops

    .

     Paris

     is

     also

     the

     birth

    place

     of

     various

     significant

     events

     and

     figures

    ,

     including

     Louis

     XIV

    ,

     Napoleon

     Bon

    ap

    arte

    ,

     and

     Victor

     Hugo

    .

     Its

     skyline

     is

     dotted

     with

     iconic

     landmarks

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     rapidly

     changing

    .

     Here

     are

     some

     possible

     trends

     that

     may

     emerge

     in

     the

     coming

     years

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

     playing

     a

     significant

     role

     in

     healthcare

    ,

     from

     disease

     diagnosis

     and

     treatment

     to

     predictive

     analytics

     for

     patient

     care

    .

     As

     AI

     becomes

     more

     advanced

    ,

     we

     may

     see

     even

     more

     widespread

     use

     in

     the

     field

    .
    


    2

    .

     AI

     in

     finance

    :

     AI

     is

     being

     used

     in

     finance

     to

     improve

     risk

     management

     and

     fraud

     detection

    .

     AI

    -powered

     algorithms

     can

     analyze

     large

     amounts

     of

     data

     to

     identify

     potential

     risks

     and

     make

     investment

     decisions

    .
    


    3

    .

     AI

     in

     manufacturing

    :

     AI

     is

     being

     used

     in

     manufacturing

     to

     optimize

     production

     processes

     and

     improve

     quality

     control

    .

     AI

     algorithms

    



```python
llm.shutdown()
```
