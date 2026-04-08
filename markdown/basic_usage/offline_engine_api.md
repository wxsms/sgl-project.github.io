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


    2026-04-08 06:52:28.068 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:52:28] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:52:28.069 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:52:28] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:52:28.069 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:52:28] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:52:28.069 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:52:28] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 06:52:28.069 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 06:52:28] Persistent cache disabled, using in-memory JIT cache


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.52it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.51it/s]


    2026-04-08 06:52:30,603 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 06:52:30] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:42,  2.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:42,  2.86s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:42,  2.86s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:42,  2.86s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:42,  2.86s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.22it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.22it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.22it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:03<00:23,  2.22it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:03<00:23,  2.22it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:03<00:23,  2.22it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:03<00:23,  2.22it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:08,  5.85it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:08,  5.85it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:08,  5.85it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:08,  5.85it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:08,  5.85it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:08,  5.85it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:08,  5.85it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:08,  5.85it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:08,  5.85it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 11.85it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 11.85it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 11.85it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 11.85it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 11.85it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 11.85it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 11.85it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 11.85it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 17.67it/s]

    Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 17.67it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 20.03it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 25.04it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 25.04it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 25.04it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 25.04it/s]

    Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 25.04it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 25.04it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 25.04it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 30.52it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 30.52it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 34.45it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 34.45it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 34.45it/s]

    Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 34.45it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 34.45it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 34.45it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 34.45it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 34.45it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 34.45it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:03, 15.04it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 15.04it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 15.04it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:03, 15.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.67it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.67it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  14%|█▍        | 8/58 [00:00<00:02, 23.67it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  19%|█▉        | 11/58 [00:00<00:03, 12.56it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  19%|█▉        | 11/58 [00:00<00:03, 12.56it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  19%|█▉        | 11/58 [00:00<00:03, 12.56it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  19%|█▉        | 11/58 [00:00<00:03, 12.56it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  19%|█▉        | 11/58 [00:00<00:03, 12.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  19%|█▉        | 11/58 [00:00<00:03, 12.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.29it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.29it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.29it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.29it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.29it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  28%|██▊       | 16/58 [00:00<00:02, 19.29it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.06it/s]Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  36%|███▌      | 21/58 [00:00<00:01, 25.06it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.06it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.06it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.06it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  36%|███▌      | 21/58 [00:01<00:01, 25.06it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.36it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.36it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.36it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.36it/s]

    Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.36it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  45%|████▍     | 26/58 [00:01<00:01, 29.36it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.25it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.21it/s]

    Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  71%|███████   | 41/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  71%|███████   | 41/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  71%|███████   | 41/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  71%|███████   | 41/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  71%|███████   | 41/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  71%|███████   | 41/58 [00:01<00:00, 38.58it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.64it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.64it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.64it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.64it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.64it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  79%|███████▉  | 46/58 [00:01<00:00, 39.64it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.73it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.73it/s]

    Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.08it/s] Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 31.49it/s]


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
    Generated text:  Mike. I am a little nervous in the new classroom. I have never been in a classroom like this. I feel like my teacher is saying things that I don't understand, like "I'm not going to be able to understand" or "I can't understand it". My teacher is quite strict and doesn't let me take notes. I'm afraid of doing something wrong because of the teacher's strictness. What should I do? A. Find someone to help me. B. Ask my parents to come to the classroom. C. Tell my teacher I don't understand. D. Tell my parents I can't understand.
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  paid 200,000 percent of his salary in tips. If he gets $x in tips per day, how much money does he make per week?
    If we know the answer to the above question, what is the value of unknown variable x?
    We are given that the president of the United States is paid 200,000 percent of his salary in tips.
    To find out how much money he makes per week, we need to calculate the total amount of money he makes from tips.
    We know that he gets $x in tips per day, so to find out how much he makes per week
    ===============================
    Prompt: The capital of France is
    Generated text:  located on the ( ) of the country.
    A. Upper left
    B. Upper right
    C. Lower left
    D. Lower right
    Answer:
    B
    
    For a certain project, the investment estimate is 20 million yuan, the construction period is 2 years, and the annual output is 2 million yuan. The total economic benefits for the project are 6 million yuan. What is the unit price of the project?
    A. 20 million yuan
    B. 10 million yuan
    C. 30 million yuan
    D. 12 million yuan
    Answer:
    B
    
    Which of the following
    ===============================
    Prompt: The future of AI is
    Generated text:  highly dependent on the collaboration of many different entities. As we continue to develop and apply AI, these entities need to adapt in a timely manner to ensure that their AI systems are as effective as possible. However, there are many challenges that will need to be overcome to achieve this goal. One of the biggest challenges is ensuring that the AI systems are not only effective but also ethical.
    AI systems are often developed with the goal of helping humans and solving problems in various domains such as healthcare, finance, and transportation. However, these systems can also be used to perpetuate biases and discrimination. It is important that the AI systems developed and used by


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age], [gender], [nationality], [occupation], and I have [number] years of experience in [field of work]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I enjoy [hobby/interest], and I'm always looking for new ways to expand my skills and knowledge. What's your favorite book or movie? I love [book/movie], and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a cultural and economic center, with a rich history dating back to the Middle Ages and a modern city that is home to many international institutions and organizations. Paris is a popular tourist destination, with millions of visitors each year, and is known for its food, fashion, and art. It is also a major hub for business and finance, with many multinational corporations and financial institutions headquartered in the city. Overall, Paris is a vibrant and dynamic city that is a must-visit for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are already being used in a wide range of applications, from self-driving cars to personalized medicine to fraud detection. As these technologies continue to improve, we can expect to see even more innovative applications emerge, from autonomous robots to virtual assistants to intelligent transportation systems. Additionally, there is a growing trend towards developing AI that is more ethical and transparent, with greater emphasis on privacy and security. This will likely lead to more robust and secure AI systems that can be trusted to operate in a safe and responsible manner. Finally, there is
    


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
    Generated text:  [name] and I'm a [occupation]. I have been working in the industry for [number of years] and have [mention any specific projects or achievements related to your profession]. My [number of years] of experience in the industry has allowed me to [mention any skills or expertise related to your profession]. I am passionate about [mention your current interests or hobbies], and I believe that my work should be [mention a positive attribute or trait related to your profession]. I am always seeking out new challenges and opportunities to [mention how you hope to contribute to the industry in the future]. Thank you for having me. (end self
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city known as the "City of Light" and a UNESCO World Heritage Site. It is the seat of government, administration, and cultural life for the country. France's capital city is located in the south of the country and is surrounded by the Rhône River. The city boasts a rich cultural heritage, including art, music, and cuisine. Paris has a population of about 2. 3 million people and is considered the fourth-largest city in the world. The city is a hub for international business, trade, and tourism.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain and will likely continue to evolve and change over the coming decades. Here are some possible trends that could emerge:
    
    1. Increased automation: As AI becomes more advanced and widespread, there will be a higher chance of automation of many tasks that are currently done by humans. This could lead to the creation of new forms of work, such as "collaborative AI" where workers work alongside machines to complete tasks.
    
    2. AI ethics: As AI becomes more advanced and integrated into our lives, there will likely be more discussion and debate around AI ethics. This could lead to a new generation of ethical guidelines for AI development, and could


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

    Your

     Name

    ]

     and

     I

     am

     a

     [

    Your

     Profession

    ]

     [

    Your

     Current

     Job

     Title

    ].

     I

     have

     over

     [

    number

    ]

     years

     of

     experience

     in

     [

    Your

     Profession

    ],

     and

     I

     have

     [

    number

    ]

     years

     of

     experience

     in

     [

    Your

     Profession

    ]

     before

     moving

     to

     [

    Your

     Current

     Location

    ]

     in

     [

    Year

    ].

     I

     have

     a

     [

    number

    ]

     years

     of

     experience

     in

     [

    Your

     Profession

    ]

     and

     [

    number

    ]

     years

     of

     experience

     in

     [

    Your

     Profession

    ].

     I

     am

     currently

     [

    number

    ]

     years

     out

     of

     [

    Number

     of

     Years

    ]

     of

     experience

     in

     [

    Your

     Profession

    ],

     and

     I

     have

     always

     been

     passionate

     about

     [

    Your

     Passion

    ].

     I

     am

     [

    number

    ]

     years

     of

     age

    ,

     and

     I

     live

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     Paris

     is

     the

     largest

     city

     in

     France

     by

     population

    ,

     with

     over

     

    2

    .

    5

     million

     residents

     as

     of

     

    2

    0

    2

    1

    .

     It

     is

     known

     for

     its

     rich

     history

    ,

     iconic

     landmarks

    ,

     and

     vibrant

     culture

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     landmarks

    ,

     including

     Notre

    -D

    ame

     Cathedral

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     an

     important

     center

     for

     politics

    ,

     art

    ,

     and

     culture

    ,

     and

     is

     an

     important

     gateway

     to

     the

     country

    's

     vast

     territories

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     a

     growing

     trend

     towards

     more

     sophisticated

     and

     nuanced

     AI

     that

     can

     learn

     and

     adapt

     from

     its

     experiences

    ,

     rather

     than

     relying

     solely

     on

     predefined

     algorithms

     and

     rules

    .

     Some

     potential

     future

     trends

     in

     AI

     include

    :
    


     

     

    1

    .

     Increased

     use

     of

     artificial

     intelligence

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     assist

     doctors

     in

     diagn

    osing

     and

     treating

     diseases

    ,

     and

     it

     has

     the

     potential

     to

     revolution

    ize

     the

     field

     of

     medicine

     by

     improving

     patient

     outcomes

     and

     reducing

     costs

    .


     

     

    2

    .

     More

     advanced

     forms

     of

     AI

    :

     AI

     is

     expected

     to

     become

     more

     powerful

     and

     capable

     of

     performing

     tasks

     that

     were

     once

     thought

     impossible

    ,

     such

     as

     natural

     language

     processing

    ,

     robotics

    ,

     and

     autonomous

     vehicles

    .


    



```python
llm.shutdown()
```
