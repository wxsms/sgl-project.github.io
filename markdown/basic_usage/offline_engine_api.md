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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.11it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.10it/s]


    2026-04-07 16:28:19,113 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-07 16:28:19] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:32,  2.67s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.77it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:02<00:03, 12.21it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:02<00:03, 12.21it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.21it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.21it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.21it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.21it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.21it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.21it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.40it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 25.19it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 25.19it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 25.19it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 25.19it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 25.19it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 25.19it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 25.19it/s]

    Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 30.27it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 30.27it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 30.27it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 30.27it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 30.27it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 30.27it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 30.27it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.74it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.74it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.74it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.74it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.74it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.74it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.74it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.91it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=133.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=133.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=133.21 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=7168 avail_mem=133.20 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=6656 avail_mem=133.20 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=133.20 GB):   3%|▎         | 2/58 [00:00<00:02, 18.77it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=133.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=5632 avail_mem=133.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=5120 avail_mem=133.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=4608 avail_mem=133.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=133.20 GB):   9%|▊         | 5/58 [00:00<00:02, 21.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=133.20 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.30it/s]Capturing num tokens (num_tokens=3840 avail_mem=133.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.30it/s]Capturing num tokens (num_tokens=3584 avail_mem=133.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.30it/s]Capturing num tokens (num_tokens=3328 avail_mem=133.19 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.30it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=133.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=133.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=133.18 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=133.18 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=133.17 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=133.17 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1792 avail_mem=133.17 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=133.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.04it/s]Capturing num tokens (num_tokens=1536 avail_mem=133.16 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.76it/s]Capturing num tokens (num_tokens=1280 avail_mem=133.16 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.76it/s]Capturing num tokens (num_tokens=1024 avail_mem=133.14 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.76it/s]Capturing num tokens (num_tokens=960 avail_mem=133.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.76it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=133.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.76it/s]Capturing num tokens (num_tokens=832 avail_mem=133.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.76it/s]Capturing num tokens (num_tokens=832 avail_mem=133.15 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.92it/s]Capturing num tokens (num_tokens=768 avail_mem=133.14 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.92it/s]Capturing num tokens (num_tokens=704 avail_mem=133.14 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.92it/s]Capturing num tokens (num_tokens=640 avail_mem=133.14 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.92it/s]Capturing num tokens (num_tokens=576 avail_mem=133.14 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.92it/s]Capturing num tokens (num_tokens=512 avail_mem=133.12 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.92it/s]Capturing num tokens (num_tokens=512 avail_mem=133.12 GB):  50%|█████     | 29/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=480 avail_mem=133.14 GB):  50%|█████     | 29/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=448 avail_mem=133.14 GB):  50%|█████     | 29/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=416 avail_mem=133.14 GB):  50%|█████     | 29/58 [00:00<00:00, 42.10it/s]

    Capturing num tokens (num_tokens=384 avail_mem=133.13 GB):  50%|█████     | 29/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=352 avail_mem=133.13 GB):  50%|█████     | 29/58 [00:00<00:00, 42.10it/s]Capturing num tokens (num_tokens=352 avail_mem=133.13 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=320 avail_mem=133.12 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=288 avail_mem=133.12 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=256 avail_mem=133.12 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=240 avail_mem=133.12 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=224 avail_mem=133.11 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.69it/s]Capturing num tokens (num_tokens=224 avail_mem=133.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=208 avail_mem=133.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=192 avail_mem=133.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=176 avail_mem=133.11 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.57it/s]

    Capturing num tokens (num_tokens=160 avail_mem=133.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=144 avail_mem=133.10 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.57it/s]Capturing num tokens (num_tokens=144 avail_mem=133.10 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.38it/s]Capturing num tokens (num_tokens=128 avail_mem=133.10 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.38it/s]Capturing num tokens (num_tokens=112 avail_mem=133.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.38it/s]Capturing num tokens (num_tokens=96 avail_mem=133.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.38it/s] Capturing num tokens (num_tokens=80 avail_mem=133.09 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.38it/s]Capturing num tokens (num_tokens=64 avail_mem=133.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.38it/s]Capturing num tokens (num_tokens=64 avail_mem=133.08 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.53it/s]Capturing num tokens (num_tokens=48 avail_mem=133.08 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.53it/s]Capturing num tokens (num_tokens=32 avail_mem=133.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.53it/s]Capturing num tokens (num_tokens=28 avail_mem=133.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.53it/s]

    Capturing num tokens (num_tokens=24 avail_mem=133.07 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.53it/s]Capturing num tokens (num_tokens=20 avail_mem=133.06 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.53it/s]Capturing num tokens (num_tokens=20 avail_mem=133.06 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=16 avail_mem=133.06 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=12 avail_mem=133.06 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=8 avail_mem=133.06 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.97it/s] Capturing num tokens (num_tokens=4 avail_mem=133.05 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.97it/s]Capturing num tokens (num_tokens=4 avail_mem=133.05 GB): 100%|██████████| 58/58 [00:01<00:00, 40.72it/s]


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
    Generated text:  Jakob. I am a German student in the BSc in Philosophy program at the University of Cologne. I have been studying philosophy since my first semester and I am looking forward to a fruitful future as a philosopher and an economist. I am a huge fan of music and I am passionate about my personal side, my work, and my friends. Before coming to Germany, I studied law at the University of Cologne and got a degree in economy. I have a passion for teaching English to Germans, and I love the culinary world of Belgium. I love movies and anime. I love music. I love to play the guitar and I love hiking
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the country. He is the leader of the country. He is the most important person in the country. He is the boss of the country. He is the boss of the country and he always works hard to make the country better. He is the leader of the country and he always leads the country. He is the leader of the country and he always works hard to make the country better. He is the boss of the country and he always leads the country. He is the leader of the country and he always works hard to make the country better. He is the leader of the country and he always leads the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Strasbourg
    C. Lyon
    D. Nancy
    
    To determine the capital of France, let's analyze the options step by step:
    
    A. Paris - While Paris is the capital of France, it is not the only capital city. The other options are not capitals but are important cities in France.
    
    B. Strasbourg - Strasbourg is a city in France, but it is not the capital city. The capital of France is not a city, but rather a region with its own capital city.
    
    C. Lyon - Lyon is a city in France, but it is not the capital. The capital of
    ===============================
    Prompt: The future of AI is
    Generated text:  in data, and the cloud is where most of the data is located. Many companies have deployed their data, or more specifically their data lakes, on the cloud, such as Amazon Web Services (AWS), Microsoft Azure, Google Cloud, and others. However, with all this data coming in, it's inevitable that there will be some data that is lost or corrupted in transit or on cloud storage. The data that is lost and destroyed is what we refer to as “data loss,” and it can lead to serious consequences. Some of the common causes of data loss are the storage of data on a cloud account or on the internal hard drives


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to learn more about your career and interests. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and restaurants. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French language. It is also home to many notable French artists and writers, including Pablo Picasso and Vincent van Gogh. Paris is a vibrant and dynamic city with a rich cultural and artistic heritage. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn from and adapt to human behavior and decision-making processes.
    
    2. Enhanced privacy and security: As AI becomes more advanced, there will be an increased need for privacy and security measures to protect the data and personal information that AI systems collect and process.
    
    3. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations and the responsible use of AI systems.
    
    4. Increased use of AI in healthcare: AI
    


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
    Generated text:  [Name], and I'm an AI language model. I'm here to assist you in answering any questions you have to the best of my abilities. I'm constantly learning and improving, and I'm here to help you improve your communication skills and overall understanding of the world around us. Let me know if you have any questions or if there's anything specific you'd like to learn about. I'm here to help! 📚🌐📚✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France, located in the central part of the country in the region of Occitania, which is also known as Paris. It is the 15th-largest city in the world by population, and is the seat of the French government, a country of historical significance and cultural importance. The city is known for its rich history, art, cuisine, and world-renowned architecture, and is a major tourist destination. Paris is also one of the most cosmopolitan cities in the world, known for its diverse and multicultural population. The city is home to some of the world's most iconic landmarks,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a combination of the following trends:
    
    1. Deep Learning: As the power of AI technology continues to increase, the ability of machines to learn and make decisions will continue to improve. This will allow AI systems to solve complex problems and make predictions more accurately than ever before.
    
    2. Natural Language Processing: As AI systems become more complex, they will be able to understand and respond to natural language inputs. This will allow AI to interact with people in new and innovative ways, such as through voice assistants and chatbots.
    
    3. Biometrics: Biometrics will become more prevalent in AI applications, such as biometric identification systems and


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

    name

    ].

     I

    'm

     a

     [

    character

     type

    ]

     who

     has

     always

     been

     passionate

     about

     [

    occupation

    ].

     I

     believe

     that

     [

    reason

     for

     being

    ]

     and

     [

    mot

    ivation

     for

     being

    ]

     is

     my

     calling

    .
    


    I

     enjoy

     [

    reason

     for

     being

    ]

     because

     [

    why

     it

    's

     fun

    ].

     I

     believe

     in

     [

    why

     I

     believe

    ].

     I

    'm

     constantly

     growing

    ,

     learning

    ,

     and

     trying

     new

     things

     to

     [

    why

     I

    'm

     continually

     improving

    ].

     I

    'm

     committed

     to

     [

    why

     I

    'm

     always

     evolving

    ],

     and

     I

     enjoy

     [

    why

     I

     never

     tire

     of

     doing

     so

    ].
    


    I

     enjoy

     [

    reason

     for

     being

    ]

     because

     [

    why

     it

    's

     fun

    ].

     I

     believe

     in

     [

    why

     I

     believe

    ].

     I

    'm

     constantly

     growing

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     French

     capital

     city

     is

     Paris

    .

     It

     is

     the

     largest

     city

     in

     France

     and

     the

     most

     populous

     city

     in

     the

     European

     Union

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     art

    ,

     culture

    ,

     and

     cuisine

    .

     Paris

     is

     also

     home

     to

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     many

     other

     iconic

     landmarks

     and

     attractions

    .

     The

     city

     is

     a

     cultural

     hub

     and

     a

     world

    -ren

    owned

     tourist

     destination

    ,

     and

     it

     plays

     a

     crucial

     role

     in

     shaping

     French

     identity

     and

     politics

    .

     It

     is

     the

     official

     capital

     of

     France

     and

     is

     considered

     a

     city

    -state

    .

     Its

     name

     in

     French

     is

     "

    P

    é

    rou

    "

     meaning

     "

    place

     of

     the

     sea

    ."

     Paris

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     three

     trends

    :

     automation

    ,

     deep

     learning

    ,

     and

     ethical

     AI

    .

     
    


    Automation

     is

     likely

     to

     continue

     to

     increase

     as

     more

     tasks

     become

     automated

     by

     AI

    .

     This

     could

     lead

     to

     increased

     efficiency

    ,

     but

     it

     could

     also

     lead

     to

     job

     displacement

     for

     some

     individuals

    .

     The

     speed

     of

     automation

     is

     expected

     to

     accelerate

     in

     the

     coming

     years

    ,

     with

     more

     tasks

     becoming

     automated

    .
    


    Deep

     learning

     is

     likely

     to

     continue

     to

     dominate

     AI

     as

     the

     technology

     advances

     and

     becomes

     more

     reliable

    .

     This

     could

     lead

     to

     even

     more

     complex

     and

     sophisticated

     AI

     systems

    ,

     but

     it

     could

     also

     lead

     to

     new

     forms

     of

     AI

    ,

     such

     as

     super

    intelligence

    .
    


    Eth

    ical

     AI

     is

     likely

     to

     become

     more

     prevalent

     as

     more

    



```python
llm.shutdown()
```
