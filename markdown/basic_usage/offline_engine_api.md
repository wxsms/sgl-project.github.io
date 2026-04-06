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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.23it/s]


    2026-04-06 04:34:50,637 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 04:34:50] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.78it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:02<00:03, 12.24it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:02<00:03, 12.24it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:02<00:03, 12.24it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.24it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.24it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.24it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.24it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.24it/s]

    Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.45it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.45it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.45it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.45it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.45it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.45it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.45it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 23.05it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 23.05it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 23.05it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 23.05it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 23.05it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 23.05it/s]

    Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 23.05it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 28.78it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 28.78it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 28.78it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 28.78it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 28.78it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 28.78it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 28.78it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 32.47it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 32.47it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 32.47it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 32.47it/s]

    Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 32.47it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 32.47it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 32.47it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 36.35it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 36.35it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 36.35it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 36.35it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 36.35it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 36.35it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 36.35it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 36.35it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 36.35it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 45.20it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 45.20it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.72it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.63 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:02, 18.84it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=131.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.69it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.43it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.43it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.43it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.86it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.86it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.86it/s]Capturing num tokens (num_tokens=960 avail_mem=131.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.86it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=131.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.86it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.86it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=768 avail_mem=131.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=704 avail_mem=131.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.83it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  50%|█████     | 29/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=480 avail_mem=131.57 GB):  50%|█████     | 29/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  50%|█████     | 29/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  50%|█████     | 29/58 [00:00<00:00, 41.87it/s]

    Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  50%|█████     | 29/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  50%|█████     | 29/58 [00:00<00:00, 41.87it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=288 avail_mem=131.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=224 avail_mem=131.54 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=224 avail_mem=131.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.19it/s]

    Capturing num tokens (num_tokens=160 avail_mem=131.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.19it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s] Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=64 avail_mem=131.51 GB):  76%|███████▌  | 44/58 [00:01<00:00, 45.00it/s]Capturing num tokens (num_tokens=64 avail_mem=131.51 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=48 avail_mem=131.51 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=28 avail_mem=131.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.17it/s]

    Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.17it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.63it/s]Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.63it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.63it/s]Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.63it/s] Capturing num tokens (num_tokens=4 avail_mem=131.48 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.63it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 40.41it/s]


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
    Generated text:  Audrey. I'm a 7th grader. I love to listen to music. My favorite music genre is rock. I have three friends in my class - Becky, Alice, and Mike. Becky and Alice both like rock music very much. But they don't like me very much. I don't have a music teacher and I have to learn to play the guitar. This is very hard. Alice says she will take the guitar lessons. She is very kind and helpful to me. But she is only 11 years old. I can't have her. I'm too shy. I don't like boys. I'm
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. When he is in power, he is not allowed to do anything in private. However, when he is not in power, he can do anything in public. If he is not in power, he can do many things in private. If he is in power, he is not allowed to do anything in private. 
    
    If a person is in power, they are allowed to do anything in private, and if a person is not in power, they are not allowed to do anything in private. If someone is in power, they have more freedom of speech than someone not in power. However, if someone is not
    ===============================
    Prompt: The capital of France is
    Generated text: 
    A. Paris
    B. London
    C. Brussels
    D. Madrid
    The answer is A. Paris. Paris is the capital and most populous city of France. It is known for its famous landmarks, such as the Eiffel Tower and Notre-Dame Cathedral. The other options (London, Brussels, and Madrid) are not capitals of France. Paris is located in the Loire Valley, which is part of the Paris region, which is a major industrial and cultural hub in France. 
    
    Therefore, the correct answer is A. Paris. 
    
    (Note: The other options listed (London, Brussels, and Madrid) are
    ===============================
    Prompt: The future of AI is
    Generated text:  what we want, and even more, we hope it is what we want. In other words, we are not as concerned about the current state of AI or the most recent developments in the field as much as we are about the future.
    We strive to create an AI that can generate content in the most natural and effective way possible. The goal is to produce a level of creativity and innovation that exceeds the capabilities of any previous AI.
    This will be achieved through a combination of human expertise and artificial intelligence (AI), with a particular focus on developing a system that can generate human-like content. This will involve a combination of natural language processing (


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


    Generated text:  [Name] and I'm a [occupation] with [number of years] years of experience in [field]. I'm a [type of person] and [reason for being] [reason for being]. I'm [age] years old and [occupation] with [number of years] years of experience in [field]. I'm a [type of person] and [reason for being] [reason for being]. I'm [age] years old and [occupation] with [number of years] years of experience in [field]. I'm a [type of person] and [reason for being] [reason for being].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major economic and political center in Europe. It is also known for its fashion industry and is home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of art, culture, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations. This could lead to more sophisticated and nuanced AI that can better understand and respond to human emotions and behaviors.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, transparency, and accountability. This will require developers to be more mindful of the potential impact of their AI systems on society and to take steps to ensure that their systems are
    


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
    Generated text:  [name] and I'm a [character] who has been [description of your character] for [number of years]. If you could ever meet me, what would you like to know about me? [name]: As an AI language model, I don't have personal experiences, emotions, or a physical body. However, I can provide you with answers and insights based on the information and data you provide me with. So, feel free to share your experiences, challenges, or interests, and I'll do my best to answer your questions and provide you with valuable information! Let's chat! 😊✨
    
    Hey there! I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The answer to the riddle is Paris. Here's the concise factual statement: Paris is the capital city of France. 
    
    However, if you're looking for a brief answer that captures the essence of Paris in a single word or phrase, you might say: "The City of Light." This captures the city's iconic status and the lightness of its coat of arms. 
    
    If you're looking for a more literal answer, here it is: "The Big Bongo." This is a pun on "big city," referring to the lively atmosphere and grandeur of Paris. 
    
    If you want to say something more descriptive,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid progress and innovation, as well as a number of emerging trends. Here are some possible future trends in AI:
    
    1. Increasing Integration of AI into Various Industries: With the increasing demand for automation and efficiency in industries, AI will become more integrated into various sectors. This includes healthcare, finance, transportation, manufacturing, and manufacturing, to name a few.
    
    2. Advancements in AI Ethics and Privacy: As AI becomes more prevalent in our lives, there will be increased scrutiny on the ethical and privacy implications of AI systems. Governments and organizations will need to develop policies and standards to ensure that AI systems are used responsibly


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

     __

    ________

     and

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     country

     and

     has

     a

     rich

     history

     and

     cultural

     heritage

    .

     Paris

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     It

     is

     also

     famous

     for

     its

     fashion

     industry

     and

     its

     annual

     fashion

     week

    ,

     and

     for

     its

     cosm

    opolitan

     atmosphere

    .

     Paris

     is

     a

     major

     economic

     and

     cultural

     center

    ,

     hosting

     a

     wide

     range

     of

     world

    -ren

    owned

     events

     and

     attractions

    .

     The

     city

    's

     architecture

     and

     cuisine

     are

     also

     highly

     regarded

    .

     Overall

    ,

     Paris

     is

     a

     bustling

     and

     vibrant

     met

    ropolis

     with

     a

     unique

     blend

     of

     history

    ,

     culture

    ,

     and

     modern

    ity

    .

     Its

     status

     as

     the

     capital

     of

     France

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     different

     trends

     and

     developments

    .

     Here

     are

     some

     of

     the

     most

     likely

     trends

     to

     occur

     in

     the

     coming

     years

    :
    


    1

    .

     AI

     will

     continue

     to

     become

     more

     pervasive

     and

     integrated

     into

     our

     lives

    .

     This

     will

     likely

     involve

     applications

     in

     fields

     such

     as

     healthcare

    ,

     education

    ,

     transportation

    ,

     and

     entertainment

    .
    


    2

    .

     AI

     will

     become

     more

     general

    -purpose

    ,

     meaning

     that

     it

     will

     be

     able

     to

     solve

     a

     wide

     range

     of

     problems

    ,

     from

     natural

     language

     processing

     to

     decision

    -making

     in

     complex

     systems

    .
    


    3

    .

     AI

     will

     become

     more

     ethical

     and

     transparent

    ,

     with

     greater

     attention

     paid

     to

     the

     ethical

     implications

     of

     AI

     in

     different

     applications

    .
    


    4

    .

     AI

     will

     become

     more

     diverse

     and

     inclusive

    



```python
llm.shutdown()
```
