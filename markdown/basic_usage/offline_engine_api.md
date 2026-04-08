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


    2026-04-08 03:53:03.915 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 03:53:03] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 03:53:03.915 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 03:53:03] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 03:53:03.915 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 03:53:03] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 03:53:03.915 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 03:53:03] Persistent cache disabled, using in-memory JIT cache
    2026-04-08 03:53:03.915 DEBUG Persistent cache disabled, using in-memory JIT cache
    [2026-04-08 03:53:03] Persistent cache disabled, using in-memory JIT cache


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.29it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.29it/s]


    2026-04-08 03:53:06,442 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 03:53:06] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.90it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  5.19it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  5.19it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  5.19it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:09,  5.19it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:09,  5.19it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:09,  5.19it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:09,  5.19it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:02<00:09,  5.19it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:02<00:09,  5.19it/s]Compiling num tokens (num_tokens=1792):  16%|█▌        | 9/58 [00:02<00:09,  5.19it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:02<00:03, 12.49it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:02<00:03, 12.49it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:02<00:03, 12.49it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.49it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.49it/s] 

    Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.49it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.49it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:03<00:01, 17.66it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.61it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.61it/s]

    Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.61it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.61it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.61it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.61it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 25.89it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 25.89it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 25.89it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 25.89it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 25.89it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 25.89it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 29.97it/s]

    Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 29.97it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 29.97it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:03<00:00, 37.19it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:03<00:00, 37.19it/s]

    Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:03<00:00, 47.93it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:03<00:00, 47.93it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:03<00:00, 47.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.85it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.76 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 18.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 18.24it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 18.24it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:03, 18.24it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.51it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.75it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.75it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.69 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.75it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.75it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.75it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.75it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.75it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.36it/s]Capturing num tokens (num_tokens=448 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.58it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.58it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.58it/s]Capturing num tokens (num_tokens=352 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.58it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.58it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.58it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.72it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.72it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.72it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.72it/s]

    Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  62%|██████▏   | 36/58 [00:01<00:00, 46.72it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  71%|███████   | 41/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 29.67it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=144 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 29.67it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 25.41it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 25.41it/s]

    Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 25.41it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 25.41it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 25.41it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.20it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.20it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.20it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.20it/s]Capturing num tokens (num_tokens=24 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.20it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.20it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 27.20it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.00it/s]Capturing num tokens (num_tokens=12 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.00it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.00it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 33.00it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 33.47it/s]


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
    Generated text:  Little Fox, a cat who loves to read books. I have a favorite book of "Peter Pan" by J.M. Barrie. It's a classic that has captivated me since I first saw it when I was a little girl. What a wonderful story! Can you tell me about the plot of "Peter Pan"? Sure, Peter Pan is a magical, adventure-filled story that follows a young boy named Jack on a daring adventure to a new island. Jack befriends a magical, flying pig named Wendy, and they embark on a series of thrilling quests to find the Lost Treasure of Neverland. Along the way, they encounter
    ===============================
    Prompt: The president of the United States is
    Generated text:  a four-term President, serving two terms. How many years of public service does he have?
    To determine how many years of public service the president of the United States has, we need to understand that the president serves two full terms of office, each of which is four years long. Therefore, the total number of years of public service the president has is calculated by multiplying the number of full terms by the length of each term.
    
    Here are the steps:
    
    1. Identify the number of full terms the president serves. According to the problem, the president serves two full terms.
    2. Identify the length of each term. Each term is four
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Berlin
    C. Moscow
    D. London
    Answer:
    A
    
    Which of the following statements about the skeletal muscles of the human body is correct?
    A. Most of the skeletal muscles are classified as voluntary muscles.
    B. The muscles innervated by the brachial plexus have a relatively longer muscle belly.
    C. Skeletal muscles can produce contractions of the same length and magnitude.
    D. Skeletal muscles are the primary movers of the human body.
    Answer:
    D
    
    Which of the following statements about the human body is incorrect?
    A. The human body is the only organism on
    ===============================
    Prompt: The future of AI is
    Generated text:  bright but, as with all technologies, the road to its full potential is lined with challenges. An emerging area of AI is the study of how social and environmental systems work. It uses AI to help understand how the world works, and, in doing so, to identify areas where the technology is not yet ready to be used, and where it can potentially be used in a way that will benefit society.  The field of social media is a perfect example of this. When it comes to social media, the company Facebook is often vilified as a bullshitter. This is partly because of the algorithm of their platforms that suggests posts


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for art, music, and fashion, and is home to many world-renowned museums, theaters, and restaurants. The city is known for its romantic atmosphere and is a popular tourist destination, attracting millions of visitors each year. Paris is a city of contrasts, with its modern skyscrapers and historic neighborhoods blending seamlessly into the cityscape. It is a city of innovation
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into various industries, from manufacturing to healthcare to transportation. Automation will likely lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. AI ethics and privacy: As AI becomes more integrated into our daily lives, there will be increasing concerns about its ethical implications and potential privacy violations. There will likely be efforts to develop AI that is more transparent, accountable, and ethical.
    
    
    


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
    Generated text:  [Name] and I am [Age]. I have [Number] years of experience working in the field of [Field of work]. My expertise lies in [specific field] and I have [number of years] years of experience with [specific project or project] and a strong passion for [specific topic or subject]. I am [Number] of years old and I am passionate about [specific hobby or interest]. I am someone who is [Number] in the field of [specific field] and I am always looking for new and exciting opportunities to learn and grow. I am someone who is [Number] and I am always willing to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    Please provide a table that compares the cost of living in Paris and a neighboring city, Berlin. The table should include the following columns: City, Cost of Living, and Average Salary. Additionally, please include the average cost of housing and commuting time in both cities. Finally, please provide a comparative analysis of the housing, commuting, and cost of living metrics for Paris and Berlin.
    
    | City | Cost of Living | Average Salary | Average Housing Cost | Average Commuting Time | Average Cost of Housing | Average Commuting Time | Average Cost of Living |
    | --- | --- | --- | --- | --- | --- | --- |
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several key trends, including:
    
    1. Improved accuracy and precision: As AI technologies continue to develop, they are becoming more accurate and precise in their predictions and decisions. This will be particularly important in fields like healthcare, finance, and transportation.
    
    2. Greater integration with human decision-making: As AI becomes more integrated into everyday life, it is likely to play an even greater role in helping humans make decisions. This could include more sophisticated algorithms and machine learning that allow humans to make better-informed decisions.
    
    3. Greater use of AI in data analysis: AI will continue to be used to analyze vast amounts of data,


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

     Alex

    .

     I

    'm

     a

     friendly

    ,

     laid

    -back

     person

     who

     loves

     to

     relax

     and

     chill

     out

    .

     I

    'm

     a

     bit

     of

     a

     che

    ater

    ,

     but

     that

    's

     okay

     because

     I

    'm

     always

     trying

     to

     do

     something

     fun

     and

     exciting

     for

     others

    .

     I

     enjoy

     going

     out

     with

     friends

     and

     spending

     time

     in

     my

     living

     room

     watching

     TV

    .

     I

    'm

     also

     a

     bit

     of

     a

     leader

    ,

     but

     I

     try

     to

     take

     care

     of

     my

     own

     needs

     and

     well

    -being

     as

     well

    .

     I

    'm

     always

     looking

     for

     new

     adventures

     and

     opportunities

     to

     improve

     myself

    .

     I

     enjoy

     learning

     new

     things

     and

     trying

     new

     things

    ,

     and

     I

    'm

     always

     open

     to

     new

     experiences

    .

     I

    'm

     a

     bit

     of

     a

     dream

    er

    ,

     but

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     city

     located

     in

     the

     north

    western

     part

     of

     the

     country

    ,

     known

     for

     its

     iconic

     architecture

    ,

     diverse

     culture

    ,

     and

     vibrant

     arts

     scene

    .

     It

     is

     the

     capital

     of

     France

    ,

     the

     largest

     country

     in

     Europe

    ,

     and

     serves

     as

     the

     seat

     of

     government

     and

     capital

     of

     the

     French

     Republic

    .

     The

     city

     is

     home

     to

     many

     world

    -ren

    owned

     landmarks

    ,

     including

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

     Paris

     is

     a

     bustling

     city

     with

     a

     rich

     history

     and

     cultural

     heritage

    ,

     and

     is

     a

     major

     tourist

     destination

     for

     millions

     of

     visitors

     each

     year

    .

     The

     city

    's

     skyline

     is

     dotted

     with

     tall

     buildings

    ,

     and

     the

     city

     is

     known

     for

     its

     annual

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

    ,

     but

     some

     potential

     areas

     where

     it

     is

     likely

     to

     have

     a

     significant

     impact

     include

    :
    


    1

    .

     Autonomous

     vehicles

    :

     The

     development

     of

     self

    -driving

     cars

     is

     likely

     to

     revolution

    ize

     transportation

     and

     reduce

     accidents

    .

     AI

     algorithms

     will

     be

     used

     to

     make

     decisions

     on

     how

     to

     navigate

     the

     roads

     and

     avoid

     collisions

    .
    


    2

    .

     Smart

     homes

    :

     AI

     will

     be

     integrated

     into

     everyday

     household

     devices

    ,

     such

     as

     refriger

    ators

    ,

     washing

     machines

    ,

     and

     home

     security

     systems

    ,

     to

     improve

     efficiency

     and

     convenience

    .
    


    3

    .

     Personal

    ized

     healthcare

    :

     AI

     will

     help

     doctors

     to

     better

     understand

     patient

     data

     and

     provide

     personalized

     treatment

     plans

    .

     AI

     algorithms

     will

     also

     help

     in

     developing

     new

     medical

     treatments

    .
    


    4

    .

     Financial

     decision

    -making

    



```python
llm.shutdown()
```
