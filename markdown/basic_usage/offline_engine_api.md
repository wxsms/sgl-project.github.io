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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.79it/s]


    2026-04-27 10:22:20,460 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 10:22:20] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:36,  4.86s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.33it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:05<00:14,  3.22it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:05<00:05,  7.24it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:05<00:05,  7.24it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:05<00:05,  7.24it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:05<00:05,  7.24it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:05<00:05,  7.24it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:05<00:05,  7.24it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:05<00:05,  7.24it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:05<00:05,  7.24it/s]Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:05<00:05,  7.24it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:05<00:02, 12.19it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:05<00:02, 12.19it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:05<00:02, 12.19it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:05<00:02, 12.19it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:05<00:02, 12.19it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:05<00:02, 12.19it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:05<00:02, 12.19it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:05<00:02, 12.19it/s]Compiling num tokens (num_tokens=352):  45%|████▍     | 26/58 [00:05<00:02, 12.19it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:05<00:01, 18.27it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:05<00:01, 18.27it/s]Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:05<00:01, 18.27it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:05<00:01, 18.27it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:05<00:01, 18.27it/s]Compiling num tokens (num_tokens=224):  59%|█████▊    | 34/58 [00:05<00:01, 18.27it/s]

    Compiling num tokens (num_tokens=208):  59%|█████▊    | 34/58 [00:05<00:01, 18.27it/s]Compiling num tokens (num_tokens=192):  59%|█████▊    | 34/58 [00:05<00:01, 18.27it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 24.08it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 24.08it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 24.08it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 24.08it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 24.08it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 24.08it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 24.08it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 24.08it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 24.08it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 31.37it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 31.37it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 31.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 40.57it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.13it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=111.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=111.69 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=111.69 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]Capturing num tokens (num_tokens=7168 avail_mem=111.68 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=111.68 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]Capturing num tokens (num_tokens=6144 avail_mem=111.68 GB):   3%|▎         | 2/58 [00:00<00:03, 18.30it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=111.68 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=111.67 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=111.66 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=111.66 GB):   9%|▊         | 5/58 [00:00<00:02, 21.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=111.66 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=111.66 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=3840 avail_mem=111.66 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=3584 avail_mem=111.65 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]Capturing num tokens (num_tokens=3328 avail_mem=111.65 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.81it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=111.65 GB):  21%|██        | 12/58 [00:00<00:01, 29.56it/s]Capturing num tokens (num_tokens=3072 avail_mem=111.64 GB):  21%|██        | 12/58 [00:00<00:01, 29.56it/s]Capturing num tokens (num_tokens=2816 avail_mem=111.64 GB):  21%|██        | 12/58 [00:00<00:01, 29.56it/s]Capturing num tokens (num_tokens=2560 avail_mem=111.64 GB):  21%|██        | 12/58 [00:00<00:01, 29.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=111.64 GB):  21%|██        | 12/58 [00:00<00:01, 29.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=111.63 GB):  21%|██        | 12/58 [00:00<00:01, 29.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=111.63 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.42it/s]Capturing num tokens (num_tokens=1792 avail_mem=111.63 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.42it/s]Capturing num tokens (num_tokens=1536 avail_mem=111.63 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.42it/s]Capturing num tokens (num_tokens=1280 avail_mem=111.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.42it/s]Capturing num tokens (num_tokens=1024 avail_mem=111.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.42it/s]

    Capturing num tokens (num_tokens=960 avail_mem=111.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 34.42it/s] Capturing num tokens (num_tokens=960 avail_mem=111.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.34it/s]Capturing num tokens (num_tokens=896 avail_mem=111.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.34it/s]Capturing num tokens (num_tokens=832 avail_mem=111.61 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.34it/s]Capturing num tokens (num_tokens=768 avail_mem=111.61 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.34it/s]Capturing num tokens (num_tokens=704 avail_mem=111.61 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.34it/s]Capturing num tokens (num_tokens=640 avail_mem=111.60 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.34it/s]Capturing num tokens (num_tokens=640 avail_mem=111.60 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=576 avail_mem=111.60 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=512 avail_mem=111.59 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=480 avail_mem=111.60 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.40it/s]

    Capturing num tokens (num_tokens=448 avail_mem=111.60 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=416 avail_mem=111.60 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.40it/s]Capturing num tokens (num_tokens=416 avail_mem=111.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=384 avail_mem=111.60 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=352 avail_mem=111.59 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=320 avail_mem=111.59 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=288 avail_mem=111.58 GB):  55%|█████▌    | 32/58 [00:00<00:00, 40.73it/s]Capturing num tokens (num_tokens=256 avail_mem=111.58 GB):  55%|█████▌    | 32/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=256 avail_mem=111.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=240 avail_mem=111.58 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=224 avail_mem=111.57 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.50it/s]

    Capturing num tokens (num_tokens=208 avail_mem=111.57 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=192 avail_mem=111.57 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=176 avail_mem=111.57 GB):  64%|██████▍   | 37/58 [00:01<00:00, 41.50it/s]Capturing num tokens (num_tokens=176 avail_mem=111.57 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=160 avail_mem=111.56 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=144 avail_mem=111.56 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=128 avail_mem=111.56 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=112 avail_mem=111.56 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.17it/s]Capturing num tokens (num_tokens=96 avail_mem=111.55 GB):  72%|███████▏  | 42/58 [00:01<00:00, 42.17it/s] Capturing num tokens (num_tokens=96 avail_mem=111.55 GB):  81%|████████  | 47/58 [00:01<00:00, 42.14it/s]Capturing num tokens (num_tokens=80 avail_mem=111.55 GB):  81%|████████  | 47/58 [00:01<00:00, 42.14it/s]

    Capturing num tokens (num_tokens=64 avail_mem=111.54 GB):  81%|████████  | 47/58 [00:01<00:00, 42.14it/s]Capturing num tokens (num_tokens=48 avail_mem=111.54 GB):  81%|████████  | 47/58 [00:01<00:00, 42.14it/s]Capturing num tokens (num_tokens=32 avail_mem=111.54 GB):  81%|████████  | 47/58 [00:01<00:00, 42.14it/s]Capturing num tokens (num_tokens=28 avail_mem=111.51 GB):  81%|████████  | 47/58 [00:01<00:00, 42.14it/s]Capturing num tokens (num_tokens=28 avail_mem=111.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=24 avail_mem=111.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=20 avail_mem=111.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=16 avail_mem=111.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=12 avail_mem=111.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.86it/s]Capturing num tokens (num_tokens=8 avail_mem=111.47 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.86it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=111.47 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=4 avail_mem=111.47 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.07it/s]Capturing num tokens (num_tokens=4 avail_mem=111.47 GB): 100%|██████████| 58/58 [00:01<00:00, 38.09it/s]


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
    Generated text:  Yu Kai, a 15 year old girl from Taiwan. I have a few questions about my parents' behavior.
    
    My parents are really good friends. They make lots of fun stuff like playing games, watching movies, and playing cards. They can both be upset if things get a bit complicated. They usually both have their own opinions on many things, and they can't agree on anything.
    
    It's pretty hard to be around them for a long time, and I don't want to be around them when they're fighting.
    
    What's the best way to stop my parents from fighting?
    
    I really want my parents to get along better,
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military tanks to buy. He can either buy 500 tanks or 200 tanks a day. It costs $250 per day to operate 1 tank. He wants to know how many days would it take for the cost of operating 500 tanks to be 2500 times the cost of operating 200 tanks. How many days would it take?
    
    Let's denote the number of days it takes to buy 500 tanks as \( x \).
    
    The total cost for buying 500 tanks is \( 500 \times 2
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Given these facts, what is the capital of france? The answer is Paris. Paris is the capital of France. Does France have capital? Yes, Paris is the capital of France. Where is paris located? Paris is located in the central and northern regions of France. Is paris one of the most populous cities of France? Yes, Paris is the most populous city of France. How is paris located in terms of the country? Paris is located in the central and northern regions of France. Is paris the most populous city in france? Yes, Paris is the most populous city in France. How is paris located in terms of the
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of companies like Google. The company that owns Google's search engine has created a new program called BigQuery that makes it possible for AI to perform tasks such as learning, making decisions, and managing databases. But how big is this company? It is owned by Google, which is a subsidiary of Alphabet. It is the largest company in the world, with a market capitalization of around $2 trillion. This company is part of the tech industry and is heavily involved in various sectors such as healthcare, financial services, and cloud computing. It is also known for its work on artificial intelligence and machine learning. Google's BigQuery


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you. What can you tell me about yourself? I'm a [job title] at
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the seat of the French government and the country's cultural and political capital. Paris is a bustling metropolis with a rich history and a diverse population, making it a popular tourist destination. The city is known for its fashion, art, and cuisine, and is a major center for business, finance, and science. Paris is also home to many famous landmarks, including the Louvre, the Eiffel Tower, and the Notre-Dame Cathedral. The city is known for its romantic
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, and accountability.
    
    2. Development of more advanced hardware: As AI technology continues to advance, there will be a push to develop more powerful hardware that can handle complex tasks more efficiently.
    
    3. Integration of AI with other technologies: AI will continue to be integrated with other technologies such as blockchain, IoT,
    


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
    Generated text:  [Name] and I am a [job title or profession]. I'm excited to meet you! What brings you to this interview, and what do you bring to the table? I'm always looking for new perspectives and ideas. Thank you! 🌟✨
    
    ---
    
    This introduction is neutral and friendly, capturing the essence of a casual, conversational conversation. It sets a positive tone and shows that the character is approachable and eager to engage with the interviewer. The question of what brings you to this interview shows interest and curiosity, which are positive qualities. The use of the colon at the end reinforces that the character is excited to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a bustling city with a rich history and a large population, known for its iconic architecture, fashion, and cultural attractions.
    
    France's capital city of Paris is a bustling metropolis with a rich history and a large population. It is famous for its iconic architecture, fashion, and cultural attractions. The city is known for its landmarks like Notre-Dame Cathedral, Eiffel Tower, and the Louvre Museum. Paris is also home to many international schools and universities, making it a popular tourist destination. It's also known for its food scene, where many famous chefs and restaurants are located. Paris is a city that's
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and varied, and there are a number of trends that are likely to shape its direction. Some of the most promising trends include:
    
    1. Increased integration with human intelligence: AI is becoming more capable of performing tasks that require human-like decision-making and emotional intelligence. This integration will allow AI to interact more closely with humans, leading to more personalized and effective solutions.
    
    2. Emphasis on ethical and responsible AI: As more AI systems are developed, there will be a greater need to ensure that they are used ethically and responsibly. This will require more research and development in ethical AI principles, as well as increased regulation and oversight of AI


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

     I

     am

     a

    /an

     [

    Background

    ]

     who

     is

     [

    Age

    ].

     I

     am

     an

     artist

     that

     is

     always

     exploring

     the

     depths

     of

     my

     creativity

    .

     I

     create

     my

     art

     using

     materials

     that

     I

     can

     find

     around

     the

     house

     and

     have

     been

     practicing

     my

     art

     for

     many

     years

    .

     My

     favorite

     medium

     is

     [

    Favorite

     Medium

    ],

     but

     I

     am

     not

     limited

     to

     any

     one

     medium

    .

     I

     have

     a

     background

     in

     graphic

     design

     and

     enjoy

     using

     my

     skills

     to

     communicate

     with

     clients

    .

     I

     am

     confident

    ,

     hard

    working

    ,

     and

     always

     aim

     for

     excellence

    .

     I

     am

     excited

     to

     see

     what

     the

     future

     holds

     for

     me

    ,

     and

     am

     always

     looking

     to

     learn

     new

     things

     and

     stay

     up

    -to

    -date

     with

     the

     latest

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

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

     and

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     home

     to

     many

     historic

     sites

    ,

     including

     the

     Lou

    vre

     Museum

    ,

     which

     houses

     a

     vast

     collection

     of

     art

     and

     artifacts

     from

     all

     over

     the

     world

    .

     Paris

     is

     also

     renowned

     for

     its

     food

     and

     wine

    ,

     and

     the

     city

     has

     become

     one

     of

     the

     world

    's

     most

     famous

     culinary

     capitals

    .

     However

    ,

     with

     its

     sprawling

     urban

     spraw

    l

     and

     growing

     traffic

    ,

     Paris

     is

     currently

     facing

     challenges

     with

     increasing

     traffic

     congestion

     and

     air

     pollution

    .

     Despite

     these

     challenges

    ,

     Paris

     remains

     a

     popular

     tourist

     destination

     and

     a

     cultural

     hub

     for

     many

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     combination

     of

     technical

     advances

    ,

     changing

     societal

     needs

    ,

     and

     emerging

     technologies

     that

     will

     shape

     the

     way

     AI

     is

     used

     in

     various

     industries

     and

     applications

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     more

     people

     become

     concerned

     about

     the

     ethical

     implications

     of

     AI

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     developing

     ethical

     AI

     systems

     that

     take

     into

     account

     the

     needs

     of

     individuals

     and

     society

     as

     a

     whole

    .

     This

     may

     include

     developing

     AI

     systems

     that

     prioritize

     the

     well

    -being

     of

     people

    ,

     that

     consider

     the

     potential

     impact

     of

     AI

     on

     different

     groups

    ,

     and

     that

     take

     into

     account

     the

     long

    -term

     consequences

     of

     AI

     development

    .
    


    2

    .

     Improved

    



```python
llm.shutdown()
```
