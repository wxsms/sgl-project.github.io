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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.09it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.08it/s]


    2026-04-10 23:27:42,472 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 23:27:42] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:38,  2.78s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.80it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.54it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.54it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.54it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.81it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:03<00:03, 11.81it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.89it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.89it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.89it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.89it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.89it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.89it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.89it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.89it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.58it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.58it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.58it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.58it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.58it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.58it/s]

    Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.58it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:03<00:00, 25.58it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 32.05it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 32.05it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 32.05it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 32.05it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 32.05it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 32.05it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 32.05it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 32.05it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 38.21it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 38.21it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 38.21it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 38.21it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 38.21it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 38.21it/s]

    Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 38.21it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 38.21it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 38.21it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 46.52it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 46.52it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 46.52it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 46.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.59 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.58 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.58 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.58 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):   9%|▊         | 5/58 [00:00<00:02, 21.35it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.19it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.19it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.57 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.19it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.19it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.19it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.56 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.58it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.56 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.58it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.55 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.58it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.55 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.58it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.55 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.58it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.58it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 35.36it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 35.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.54 GB):  31%|███       | 18/58 [00:00<00:01, 35.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.52 GB):  31%|███       | 18/58 [00:00<00:01, 35.36it/s]Capturing num tokens (num_tokens=960 avail_mem=72.53 GB):  31%|███       | 18/58 [00:00<00:01, 35.36it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.53 GB):  31%|███       | 18/58 [00:00<00:01, 35.36it/s]Capturing num tokens (num_tokens=832 avail_mem=72.52 GB):  31%|███       | 18/58 [00:00<00:01, 35.36it/s]Capturing num tokens (num_tokens=832 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=768 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=704 avail_mem=72.52 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=640 avail_mem=72.51 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=576 avail_mem=72.51 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=512 avail_mem=72.50 GB):  50%|█████     | 29/58 [00:00<00:00, 40.33it/s]Capturing num tokens (num_tokens=480 avail_mem=72.52 GB):  50%|█████     | 29/58 [00:00<00:00, 40.33it/s]Capturing num tokens (num_tokens=448 avail_mem=72.52 GB):  50%|█████     | 29/58 [00:00<00:00, 40.33it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.51 GB):  50%|█████     | 29/58 [00:00<00:00, 40.33it/s]Capturing num tokens (num_tokens=384 avail_mem=72.51 GB):  50%|█████     | 29/58 [00:00<00:00, 40.33it/s]Capturing num tokens (num_tokens=352 avail_mem=72.51 GB):  50%|█████     | 29/58 [00:00<00:00, 40.33it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  50%|█████     | 29/58 [00:00<00:00, 40.33it/s]Capturing num tokens (num_tokens=320 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:00<00:00, 43.42it/s]Capturing num tokens (num_tokens=288 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:00<00:00, 43.42it/s]Capturing num tokens (num_tokens=256 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:00<00:00, 43.42it/s]Capturing num tokens (num_tokens=240 avail_mem=72.50 GB):  60%|██████    | 35/58 [00:00<00:00, 43.42it/s]Capturing num tokens (num_tokens=224 avail_mem=72.49 GB):  60%|██████    | 35/58 [00:01<00:00, 43.42it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  60%|██████    | 35/58 [00:01<00:00, 43.42it/s]Capturing num tokens (num_tokens=208 avail_mem=72.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=192 avail_mem=72.49 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.22it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=160 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=144 avail_mem=72.48 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.22it/s]Capturing num tokens (num_tokens=128 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.91it/s]Capturing num tokens (num_tokens=112 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.91it/s]Capturing num tokens (num_tokens=96 avail_mem=72.47 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.91it/s] Capturing num tokens (num_tokens=80 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.91it/s]Capturing num tokens (num_tokens=64 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.91it/s]Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  78%|███████▊  | 45/58 [00:01<00:00, 45.91it/s]Capturing num tokens (num_tokens=48 avail_mem=72.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=32 avail_mem=72.46 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.05it/s]

    Capturing num tokens (num_tokens=28 avail_mem=72.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=24 avail_mem=72.45 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=20 avail_mem=72.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=16 avail_mem=72.44 GB):  86%|████████▌ | 50/58 [00:01<00:00, 46.05it/s]Capturing num tokens (num_tokens=16 avail_mem=72.44 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=12 avail_mem=72.44 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=8 avail_mem=72.43 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.27it/s] Capturing num tokens (num_tokens=4 avail_mem=72.43 GB):  95%|█████████▍| 55/58 [00:01<00:00, 46.27it/s]Capturing num tokens (num_tokens=4 avail_mem=72.43 GB): 100%|██████████| 58/58 [00:01<00:00, 40.53it/s]


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
    Generated text:  Salamander, and I'm a male human who loves to make friends. I've been writing for years and I've even got a website. I've written several stories and short stories, and I've also started writing short scripts. What are your favorite books or movies, and why do you like them? As for my hobbies, I enjoy playing music and writing music in my free time. Can you also tell me about your music and the type of music you like to create? And what is your current project? Salamander loves to write and create music, and it shows in the tone of his posts. He's passionate
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. A( ) is a woman. A woman. (A) President. (B) President's. (C) Woman. (D) Woman's. To solve this problem, we need to determine the correct pronoun to use in the sentence. The sentence is: "A president of the United States is a man. A woman." We need to identify whether the pronoun "A" should be used before "president" or after "president."
    
    1. Identify the type of pronoun:
       - A pronoun typically refers to a noun or noun phrase that follows it.
    
    2. Determine the position
    ===============================
    Prompt: The capital of France is
    Generated text:  the _____. A. Centre B. Hauts-de-France C. Paris D. Brittany
    
    C. Paris
    
    Paris is the capital city of France. It is the largest city in the country, located on the River Seine in the Île-de-France region, and serves as the administrative center for France's three departments.
    
    The capital of the UK is the _____. A. London B. Scotland C. Scotland D. London
    
    A. London
    
    London is the capital city of the United Kingdom (UK). It is the largest and most populous city in the country, located on the River Thames in the south-central part
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain but could be one of the most transformative, empowering, and perilous of all time. With unprecedented opportunities come enormous risks. What does it mean to be an AI-empowered citizen today? Can we adapt to the challenges that lie ahead? The future of AI is essential to everyone and this must be shared and considered by all. While AI is the key to unlocking the vast potential of the future, it is also a critical challenge that requires thought, innovation, and a commitment to the greater good. As we navigate the complex and rapidly evolving world of AI, it is crucial to be thoughtful and reflective, and to embrace the promise


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill or Hobby] enthusiast. I love [What I Enjoy Doing]. I'm always looking for new challenges and opportunities to learn and grow. I'm a [What I Like to Do] person. I'm always ready to learn and improve myself. I'm a [What I Like to Do] person. I'm always looking for new challenges and opportunities to learn and grow. I'm a [What I Like to Do] person. I'm always ready to learn and improve myself. I'm a [What I Like to Do]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of the French Revolution and the seat of the French government. Paris is a bustling metropolis with a rich cultural heritage and is a popular tourist destination. The city is known for its cuisine, fashion, and art scene. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has played a significant role
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, particularly in areas such as diagnosis, treatment planning, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud detection, and investment decision-making. As
    


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
    Generated text:  [Your Name]. I'm a [Your Profession], and I'm a dedicated [Your Specialty or Field of Expertise] expert. I have [Number of Years] years of experience in this field, and I'm always looking for new and exciting challenges to tackle. I'm a [Your Interests or Personality] person, and I enjoy staying up-to-date with the latest trends and technologies in the field. I'm known for my [Number of Achievements or Achievements in the Industry] in this field. I'm excited to get started with you! Let's chat! [Your Name]. [Your Contact Information]. [Your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower and romantic art deco architecture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  full of exciting possibilities and potential challenges. Here are some possible future trends in AI:
    
    1. AI will continue to become more accessible and affordable for individuals and businesses. This could lead to a wider adoption of AI technologies, from self-driving cars to personalized healthcare.
    
    2. AI will become more integrated with natural language processing and machine learning, making it easier for machines to understand and interpret human language. This could lead to more intelligent and context-aware AI systems.
    
    3. AI will become more ethical and fair, with regulations and guidelines being developed to ensure that AI technologies are used responsibly and equitably. This could lead to a more positive impact on


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

     am

     a

     [

    occupation

    ]

     with

     [

    number

    ]

     years

     of

     experience

    .

     My

     unique

     selling

     point

     is

     my

     [

    insert

     one

     or

     two

     words

     that

     describe

     a

     particular

     skill

    ,

     talent

    ,

     or

     personality

     trait

    ],

     and

     I

     believe

     I

    'm

     the

     perfect

     [

    insert

     a

     positive

     word

     ending

     in

     "-

    ness

    "]

     for

     this

     position

    .

     I

     have

     been

     [

    insert

     at

     least

     three

     things

     you

     have

     achieved

     or

     grown

     in

     your

     career

     that

     have

     made

     you

     stand

     out

    ],

     and

     I

     am

     always

     eager

     to

     learn

     new

     things

     and

     explore

     new

     opportunities

    .

     Looking

     forward

     to

     having

     the

     opportunity

     to

     work

     with

     you

     and

     contribute

     to

     your

     success

    !

     [

    Start

     your

     introduction

     here

    ]

     Hi

     [

    Name

    ],

     I

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    What

     is

     the

     capital

     of

     France

    ?
    


    The

     capital

     of

     France

     is

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

    ,

     but

     some

     possible

     trends

     that

     could

     lead

     to

     significant

     changes

     in

     the

     field

     include

    :
    


    1

    .

     Increased

     automation

     and

     personal

    ization

    :

     As

     AI

     becomes

     more

     capable

    ,

     it

     is

     likely

     to

     be

     more

     capable

     of

     performing

     tasks

     and

     providing

     personalized

     services

     to

     users

    .

     This

     could

     lead

     to

     a

     shift

     towards

     more

     automation

     and

     personal

    ization

     in

     various

     industries

    .
    


    2

    .

     Improved

     data

     and

     machine

     learning

    :

     With

     the

     increasing

     amount

     of

     data

     available

    ,

     AI

     systems

     will

     become

     more

     sophisticated

     and

     capable

     of

     learning

     from

     data

    .

     This

     could

     lead

     to

     the

     development

     of

     more

     advanced

     and

     more

     accurate

     models

     that

     can

     handle

     complex

     and

     unpredictable

     situations

    .
    


    3

    .

     Enhanced

     ethical

     concerns

    :

     As

     AI

     becomes

     more

     integrated

     into

     daily

    



```python
llm.shutdown()
```
