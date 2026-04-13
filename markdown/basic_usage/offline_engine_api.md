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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.94it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.94it/s]


    2026-04-13 21:35:48,689 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-13 21:35:48] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:21,  2.47s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:21,  2.47s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:21,  2.47s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:21,  2.47s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:26,  2.02it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:26,  2.02it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:26,  2.02it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:26,  2.02it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:26,  2.02it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:26,  2.02it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:26,  2.02it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.16it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.81it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.81it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.81it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.81it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.81it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.81it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 13.81it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:02<00:02, 13.81it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:02<00:02, 13.81it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:02<00:01, 21.21it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:02<00:01, 21.21it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:02<00:01, 21.21it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:02<00:01, 21.21it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:02<00:01, 21.21it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:02<00:01, 21.21it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:02<00:01, 21.21it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:02<00:01, 21.21it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 21.21it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 29.24it/s]

    Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 29.24it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 35.92it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 35.92it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 35.92it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 35.92it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 35.92it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 35.92it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 35.92it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 35.92it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 42.13it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 42.13it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 42.13it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 42.13it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 42.13it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 42.13it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 42.13it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 42.13it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 42.13it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 42.13it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.66it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.66it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.66it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.66it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 23.07it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 23.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 23.07it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 23.07it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 23.07it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.31it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.31it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.31it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 27.25it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.62it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.62it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.62it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.62it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.62it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.62it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.62it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.23it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.23it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.23it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.23it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.23it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  40%|███▉      | 23/58 [00:00<00:00, 41.23it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.26it/s]

    Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.26it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.26it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.58it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.58it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.58it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.58it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.58it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 23.51it/s]Capturing num tokens (num_tokens=240 avail_mem=76.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 23.51it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 23.51it/s]

    Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 23.51it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 23.51it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 23.51it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 23.51it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.67it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.91it/s]

    Capturing num tokens (num_tokens=32 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.95it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.95it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.95it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.95it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 32.95it/s]


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
    Generated text:  Jessica. I am a high school student in Senior 2. I have a lot of friends and I love making new friends. I have a lot of friends, and I love making new friends. I like to play games and I have a lot of friends. I have a lot of friends, and I like to play games. I have a lot of friends. Jessica says "I have a lot of friends" twice in the text. How many times does Jessica say this sentence?
    
    To determine how many times Jessica says "I have a lot of friends", we need to carefully read and count the instances given in the text.
    
    1
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking to elect a new Chief Justice. This is a very important decision because it is likely to affect how the federal court system operates. The New York Times reported that in a recent election, the voter turnout was as high as 44% of the eligible voters. In 2000, the voter turnout was 19%. If we graph the voter turnout as a function of the number of years since 2000, what would be the equation of this line? To determine the equation of the line representing the voter turnout as a function of the number of years since 2000, we need to
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    Based on that paragraph can we conclude that the hypothesis "Paris is the capital of France."?
    Pick your answer from: (I) yes (II) it is not possible to tell (III) no
    (I) yes
    
    The hypothesis "Paris is the capital of France" is supported by the information provided in the paragraph. The paragraph explicitly states that "Paris is the capital of France," making the statement true and supported by the information given. Therefore, we can conclude that the hypothesis is correct. Given this information, the appropriate answer is (I) yes. The hypothesis is accurate based on the information provided in the paragraph
    ===============================
    Prompt: The future of AI is
    Generated text:  dark and uncertain, with the potential for incredible productivity gains and transformational impact. But when it comes to understanding the impact of AI on society and the economy, the conversation is far from clear. What is the future of AI and how can we mitigate its potential negative impacts? One possible way to address this question is through the use of data and machine learning techniques. By collecting, analyzing, and utilizing data from a variety of sources, businesses and organizations can gain a deeper understanding of their customers, markets, and business operations. This understanding can then be used to inform decisions and take advantage of new opportunities that are created by AI. By leveraging


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title] and improve my skills. I'm a [job title] and I'm always looking for ways to [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and music, and is a major tourist destination. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is home to many museums, art galleries, and cultural institutions, and is a hub for the arts and entertainment industry. Paris is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased automation: As AI becomes more advanced, it is likely to automate more and more tasks, freeing up human workers to focus on more complex and creative work. This could lead to a shift in the job market, with many jobs becoming automated and replaced by AI.
    
    2. Improved privacy and security: As AI becomes more advanced, it is likely to require more data to function effectively. This could lead to increased
    


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
    Generated text:  [Name], and I am a [profession/occupation]. I am an [industry] expert with over [number of years of experience] years of experience in [specific field or area of expertise]. I am [theoretical or actual] passionate about [what you do best], and I am always looking for [why you are a good fit for this job]. I have always been [what you consider to be] exceptional in [describe your areas of excellence], and I am always eager to learn and improve my skills. I enjoy [the reason I pursue this profession] and I am confident in my ability to succeed in this role.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, often referred to as "The City of Love" and "The City of Light". It is home to iconic landmarks such as Notre-Dame Cathedral, the Eiffel Tower, and the Louvre Museum. Paris is known for its vibrant arts scene, fashion industry, and culinary traditions, making it a global center of culture and commerce. The city is also home to a diverse population of French-speaking inhabitants, with Paris being the capital of 54 different departments. As of 2023, Paris has over 3 million inhabitants and is a major economic hub for Europe and beyond. French, along with English,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving, and there are numerous potential trends that will shape the industry in the coming years. Some of the most promising areas include:
    
    1. Large Language Models: With the rise of big data and artificial intelligence, the availability of large language models (LLMs) is on the rise. These models can be trained to understand and generate human-like language, which has applications in fields such as natural language processing, machine translation, and conversational AI.
    
    2. Autonomous Vehicles: Autonomous vehicles (AVs) are rapidly becoming a reality, and there are many potential future trends in this area. These vehicles will likely become more advanced and reliable,


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

    First

     Name

    ]

     and

     I

    'm

     [

    Last

     Name

    ].

     I

    'm

     a

     [

    insert

     the

     profession

     of

     the

     character

    ]

     by

     trade

    ,

     and

     I

    've

     always

     been

     passionate

     about

     [

    insert

     a

     personal

     interest

     or

     hobby

    ]

     in

     my

     free

     time

    .
    


    I

     have

     a

     keen

     interest

     in

     [

    insert

     a

     personal

     interest

     or

     hobby

    ]

     that

     has

     always

     driven

     me

     to

     pursue

     it

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    insert

     a

     personal

     interest

     or

     hobby

    ].

     I

     love

     sharing

     my

     love

     of

     [

    insert

     a

     personal

     interest

     or

     hobby

    ]

     with

     others

     and

     finding

     new

     ways

     to

     express

     myself

     creatively

    .
    


    I

    'm

     a

     [

    insert

     a

     personal

     interest

     or

     hobby

    ]

     and

     I

    've

     always

     been

     drawn

     to

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ré

    pub

    lique

    "

     and

     the

     "

    City

     of

     Light

    ."

     It

     is

     located

     on

     the

     Se

    ine

     River

     and

     is

     the

     third

     largest

     city

     in

     the

     European

     Union

    .

     Paris

     is

     known

     for

     its

     rich

     history

    ,

     diverse

     culture

    ,

     and

     iconic

     architecture

    .

     The

     city

     is

     also

     home

     to

     several

     world

    -ren

    owned

     museums

    ,

     such

     as

     the

     Lou

    vre

     and

     the

     National

     Library

     of

     France

    .

     Paris

     is

     an

     important

     center

     of

     culture

     and

     innovation

    ,

     attracting

     visitors

     from

     around

     the

     world

    .

     Additionally

    ,

     the

     city

     is

     home

     to

     a

     large

     cultural

     industry

    ,

     including

     film

    ,

     music

    ,

     and

     fashion

    ,

     which

     contributes

     to

     the

     city

    's

     cultural

     identity

    .

     It

     is

     also

     known

     for

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advances

     in

     computer

     technology

    ,

     changes

     in

     societal

     values

    ,

     and

     shifts

     in

     policy

     and

     regulation

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

     Increasing

    ly

     autonomous

     machines

    :

     As

     AI

     becomes

     more

     complex

     and

     capable

    ,

     it

    's

     likely

     that

     autonomous

     machines

     will

     become

     increasingly

     common

    ,

     with

     robots

     and

     other

     autonomous

     systems

     performing

     a

     wider

     range

     of

     tasks

     than

     they

     can

     now

     do

    .
    


    2

    .

     Improved

     integration

     with

     human

     expertise

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     may

     see

     more

     integration

     of

     AI

     with

     human

     expertise

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     education

    ,

     where

     human

     judgment

     and

     decision

    -making

     are

     often

     more

     valuable

     than

     purely

    



```python
llm.shutdown()
```
