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

    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-17 01:22:29] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.42it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.41it/s]


    2026-04-17 01:22:33,551 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-17 01:22:33] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:37,  2.76s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.67it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.67it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.67it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.67it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.67it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.74it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.60it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.60it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.60it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.60it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.60it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.60it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.60it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.60it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.60it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.60it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.51it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.51it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.51it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.51it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.51it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.51it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.51it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.51it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.51it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.51it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.51it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 50.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.50it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 17.84it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.84it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.84it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.84it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 20.81it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 20.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 20.81it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.84it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.84it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.56it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.56it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.56it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.56it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.26 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.58it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=120.24 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.58it/s] Capturing num tokens (num_tokens=896 avail_mem=120.25 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  33%|███▎      | 19/58 [00:00<00:01, 35.58it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.20it/s]Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.20it/s]Capturing num tokens (num_tokens=704 avail_mem=120.24 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.20it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.20it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.20it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  41%|████▏     | 24/58 [00:00<00:00, 38.20it/s]

    Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  50%|█████     | 29/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=480 avail_mem=120.24 GB):  50%|█████     | 29/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  50%|█████     | 29/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  50%|█████     | 29/58 [00:00<00:00, 40.01it/s]Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:00<00:00, 41.27it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.27it/s]

    Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  59%|█████▊    | 34/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  67%|██████▋   | 39/58 [00:01<00:00, 41.99it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.81it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  76%|███████▌  | 44/58 [00:01<00:00, 42.81it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  84%|████████▍ | 49/58 [00:01<00:00, 43.01it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.45it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.45it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.45it/s]

    Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.45it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  93%|█████████▎| 54/58 [00:01<00:00, 43.45it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 38.79it/s]


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
    Generated text:  Mike. I have a great job helping people solve problems. My office is in Seattle, Washington. I would like to tell you about a project that I am working on. 
    
    There is a big problem in the area of personal computers and I have a solution to it. I want you to review my project and tell me if it is a good idea to hire me. 
    
    Tell me about your company, your job, your background and your current situation.
    
    I am more interested in the project itself, not the background of your company, your job or your current situation. You are welcome to tell me anything about the project you are working
    ===============================
    Prompt: The president of the United States is
    Generated text:  a post that can be filled by anyone as long as they meet the following requirements: (1) be a citizen of the United States; (2) be at least 35 years old; (3) be a man; (4) be competent in English. Now, we randomly select 15 presidents from all the candidates. The number of ways to select 15 presidents such that at least one of them is a woman is ______ (answer with a number). To determine the number of ways to select 15 presidents such that at least one of them is a woman, we can use the complementary counting method.
    ===============================
    Prompt: The capital of France is
    Generated text:  in ( )
    A. Paris
    B. Lyon
    C. Marseille
    D. Bordeaux
    
    To determine the capital of France, we need to identify the official capital of France. The capital of France is typically the largest city in the country. The capital cities of France are as follows:
    
    1. Paris (city of light)
    2. Lille
    3. Nice
    4. Strasbourg
    5. Bordeaux
    
    Among these cities, the largest city is Paris, which is also known as the "City of Light." Therefore, the capital of France is Paris.
    
    The correct answer is: \boxed{A}
    ===============================
    Prompt: The future of AI is
    Generated text:  in full bloom. From the rise of deep learning to the proliferation of AI models, the path of AI is never-ending. This is evident by the addition of new technologies, such as AI, to the various industries such as banking, healthcare, and manufacturing. It is estimated that by 2025, AI could impact the economic output of nearly $1 trillion in value.
    Artificial Intelligence (AI) is a branch of computer science that enables machines to simulate intelligence. This includes learning, reasoning, and decision-making capabilities. AI is becoming more prevalent in various industries, from healthcare to finance, and has the potential to transform the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I have a [major] in [major]. I'm a [interest] person, and I enjoy [activities]. I'm always looking for new experiences and learning new things. What's your favorite hobby? I love [hobby], and I'm always looking for new ways to expand my skills and knowledge. What's your favorite book or movie? I love [book/movie], and I'm always looking
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, with many famous designers and fashion houses operating in the area. Paris is a bustling metropolis with a diverse population and a rich cultural heritage. It is a popular tourist destination and a major economic center in Europe. The city is also home to many international organizations and institutions, including the French Academy of Sciences and the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI will continue to automate routine tasks, freeing up human workers to focus on more complex and creative work. This will lead to a shift towards a more human-centric approach to AI development and deployment.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there will be an increased risk of data breaches and privacy violations. To address this, there will be a focus on improving privacy and security measures in AI systems.
    
    3. Enhanced human-machine collaboration: AI will continue to be used alongside human workers to enhance their capabilities. This will involve developing AI systems that can
    


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
    Generated text:  [Name] and I am [Age]. I love [What I Do for a Living] and I am passionate about [What I Enjoy Involving Myself In]. I also enjoy [What I Do For Fun] and I love [What I Do For My Family]. I am excited to meet you and learn more about you. Let's see if we can make a great connection and get to know each other better. [Name]. Your introduction is concise and to the point. It provides a clear understanding of your name, age, and occupation. However, you could consider adding some personal details, such as your hobbies, interests
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is the largest city in Europe and the 15th largest in the world. It is known for its beautiful architecture, rich history, and vibrant culture. The city is home to iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral, as well as numerous museums, cafes, and theaters. It is a UNESCO World Heritage Site and a UNESCO City of the Culture. Paris is also the seat of the French government and the capital of the French Republic. The city is home to over 10 million people and is the third-most populous metropolitan area in the world. It is an important economic and cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting, with potential applications in fields such as healthcare, finance, education, and transportation. Here are some possible future trends in AI:
    
    1. Increased precision and accuracy: As AI becomes more sophisticated, it is expected to improve its ability to analyze and interpret data more accurately. This will lead to more precise diagnoses, more accurate financial predictions, and more informed policy recommendations.
    
    2. Enhanced cognitive and decision-making abilities: AI systems are expected to become even more intelligent and able to make decisions and decisions on their own. This will enable them to operate in a wider range of contexts and tasks, making them more adaptable and efficient.
    
    3. Inter


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

    ].

     I

     am

     [

    Age

    ]

     years

     old

    .

     I

     have

     [

    occupation

    ]

     and

     [

    most

     significant

     achievement

    ].

     What

     brings

     you

     to

     this

     moment

     in

     time

    ?

     I

    'm

     here

     to

     [

    why

     you

    're

     here

    ].

     And

     what

     do

     you

     think

     about

     [

    future

    ].

     I

    'm

     intrigued

     by

     [

    future

     scenario

    ]

     because

     [

    why

     you

    're

     interested

    ].

     Lastly

    ,

     what

     brings

     you

     to

     this

     moment

     in

     time

    ?

     I

    'm

     [

    Name

    ]

     [

    Age

    ]

     years

     old

    ,

     and

     I

     came

     to

     [

    current

     location

    ].

     I

    'm

     here

     to

     [

    reason

     for

     being

     here

    ].

     And

     I

    'd

     like

     to

     learn

     more

     about

     [

    future

     scenario

    ],

     [

    reason

     you

    're

     interested

    ],

     and

     [

    future

     you

    're

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    To

     elaborate

    ,

     Paris

     is

     the

     capital

     city

     of

     the

     country

     of

     France

    .

     It

     is

     located

     on

     the

     Î

    le

     de

     la

     C

    ité

    ,

     a

     

    1

    1

    .

     

    2

    5

     km

    -long

     stretch

     of

     the

     Se

    ine

     river

    ,

     which

     flows

     through

     the

     middle

     of

     the

     city

    .

     The

     city

     is

     one

     of

     the

     largest

     cities

     in

     the

     world

    ,

     with

     an

     area

     of

     

    3

    6

    .

     

    3

    1

     km

    ²

    .

     It

     is

     known

     as

     the

     city

     of

     love

    ,

     as

     it

     is

     the

     location

     of

     the

     Lou

    vre

     Museum

    ,

     the

     most

     famous

     landmark

     in

     the

     city

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

     the

     famous

     landmark

     for

     its

     rectangular

     shape

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     an

     exciting

     and

     rapidly

     evolving

     field

    ,

     with

     many

     potential

     trends

     that

     could

     shape

     the

     development

     of

     this

     technology

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

     Increased

     integration

     with

     physical

     devices

    :

     With

     the

     development

     of

     more

     powerful

     and

     affordable

     chips

    ,

     it

     is

     possible

     that

     AI

     algorithms

     could

     be

     integrated

     directly

     into

     physical

     devices

     such

     as

     smartphones

    ,

     wear

    ables

    ,

     and

     robots

    .

     This

     would

     enable

     AI

     to

     learn

     and

     adapt

     to

     a

     wide

     range

     of

     physical

     situations

     and

     tasks

    .
    


    2

    .

     Enhanced

     cognitive

     processing

     power

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     it

     is

     likely

     that

     we

     will

     see

     more

     powerful

     AI

     algorithms

     that

     can

     process

     and

     analyze

     complex

     data

     sets

     in

     ways

     that

     were

     previously

     impossible

    .

    



```python
llm.shutdown()
```
