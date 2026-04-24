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

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-24 00:23:52] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.26it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.26it/s]


    2026-04-24 00:23:57,536 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 00:23:57] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:08,  5.80it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 13.97it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 13.97it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 13.97it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 13.97it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 13.97it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 13.97it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 13.97it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 13.97it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 13.97it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 13.97it/s]

    Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:03<00:02, 13.97it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 23.20it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 23.20it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 23.20it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 23.20it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 23.20it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 23.20it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 23.20it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 23.20it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 23.20it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 23.20it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:03<00:01, 23.20it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 33.52it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 33.52it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 33.52it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 33.52it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 33.52it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 33.52it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:03<00:00, 33.52it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:03<00:00, 33.52it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:03<00:00, 33.52it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:03<00:00, 33.52it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:03<00:00, 33.52it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 43.94it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 43.94it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 43.94it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 43.94it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 43.94it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 43.94it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 43.94it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 43.94it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 43.94it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.26it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.48it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.25it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.25it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.25it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.25it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.25it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.25it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.55it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 42.57it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 42.57it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:00<00:00, 42.57it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.57it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.57it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:00<00:00, 42.57it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:00<00:00, 42.57it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=240 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:00<00:00, 45.16it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 45.16it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  71%|███████   | 41/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 47.02it/s]

    Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 47.02it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 47.02it/s] Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  81%|████████  | 47/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=32 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.42it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.42it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.42it/s]

    Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.42it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.42it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.42it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.42it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 49.06it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 42.37it/s]


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
    Generated text:  Christian Hossel, a lifelong Austrian citizen and the chairman of the Austrian Federal Cabinet of Ministers. I am the first politician to have been appointed to the cabinet of a leader of a smaller country by a member of the Austrian coalition government. I am the first Austrian minister of transport and maritime affairs. My party is the Social Democratic Party of Austria, and I hold the role of Minister President of Austria since 2015.
    Christian Hossel was born on March 2, 1954, in Vienna, Austria. He is married to Magdalena Merkli and has two children, Maja and Aaron
    ===============================
    Prompt: The president of the United States is
    Generated text:  representing the country, and the vice president is representing the state. Among the 100 states, which state is the vice president representing?
    To determine which state the vice president is representing, we need to understand the structure of the U.S. political system. The president is elected by the U.S. Congress, and the vice president is the second-in-command to the president in the event of the president's death or resignation. This means that the vice president is always representing the state in which they are elected to the presidency.
    
    Since the president is representing the entire United States, the vice president must be representing the state that has the
    ===============================
    Prompt: The capital of France is
    Generated text:  ( )
    A. Paris
    B. Lyon
    C. Marseille
    D. Bordeaux
    Answer: A
    
    Which of the following is the primary cause of liver cirrhosis in patients with chronic hepatitis?
    A. Infection
    B. Nutritional imbalance
    C. Bile stasis
    D. Heavy alcohol consumption
    E. Metabolic disorders
    Answer: B
    
    Which of the following items does not belong to the category of energy sources?
    A. Oil
    B. Natural Gas
    C. Coal
    D. Solar Energy
    Answer: D
    
    What is the primary target of drug therapy for patients with chronic myeloid leukemia
    ===============================
    Prompt: The future of AI is
    Generated text:  increasingly clear. Over the past few years, AI has become a key part of the daily lives of many. From medical imaging and diagnostics, to speech recognition and translation, to autonomous vehicles, AI has become an integral part of our lives. In fact, it is expected that by 2025, AI will account for more than 15% of global GDP. This represents a significant shift in the way we live our lives, and one that will require a major investment in research and development to keep pace with the demand for AI solutions.
    There are many different types of AI that can be found in today’s world. These


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and art galleries. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on modern French society. Paris is also home to many famous French artists, writers, and musicians, including Pablo Picasso, Vincent van Gogh, and Claude Monet. The city is a major center for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical implications. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Greater reliance on AI for decision-making: AI is likely to become more integrated into decision-making processes, allowing machines to make more informed
    


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
    Generated text:  [Name] and I am a [Name] who works at [Company]. I am an experienced [job title] and I specialize in [description of job]. I am confident in my ability to [describe something I do well]. I am currently [Current position]. How can I assist you today?
    [Name] I'm [Name] and I work at [Company]. I specialize in [job title] and my primary focus is [job title]. I am confident in my ability to [describe something I do well]. I am currently [Current position]. How can I assist you today? [Name] What can I do for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Its landmarks include the Eiffel Tower and the Louvre Museum, and it is the largest city in the European Union. Paris is known for its romantic atmosphere, historic architecture, and vibrant nightlife. It is also a UNESCO World Heritage site. France's capital city is a major hub for international diplomacy, commerce, and culture. Its government offices and administrative centers are situated in the historic city center. The city is also home to a large number of museums, including the Musée d'Orsay, the Louvre, and the Musée d'Orsay. Paris is also a tourist destination for its stunning architecture, French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid innovation, automation, and integration with various industries. Here are some possible trends that may occur in the coming years:
    
    1. Increased automation: The use of AI and robots in manufacturing, healthcare, and other industries is expected to increase. This will lead to increased efficiency, reduced costs, and the ability to handle more complex tasks.
    
    2. Deep learning and artificial intelligence: Deep learning algorithms will continue to improve and become more sophisticated, allowing for even more complex tasks to be performed by AI systems. This will lead to new applications of AI, such as autonomous vehicles, smart homes, and personalized medicine.
    
    3.


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

     an

     [

    occupation

    ]

     with

     [

    major

     qualifications

     or

     experience

    ]

     in

     [

    specific

     area

     of

     expertise

    ].

     I

     believe

     in

     [

    values

     or

     principles

    ]

     and

     I

     am

     committed

     to

     [

    part

    icipation

     or

     contribution

     to

    ]

     [

    specific

     group

     or

     cause

    ].
    


    [

    Name

    ],

     your

     character

     is

     an

     AI

     assistant

     designed

     to

     assist

     with

     all

     your

     tasks

    .

     I

     strive

     to

     provide

     you

     with

     accurate

    ,

     helpful

    ,

     and

     timely

     responses

    .

     I

     am

     always

     here

     to

     help

     you

     with

     any

     questions

     you

     may

     have

     or

     tasks

     you

     may

     need

     help

     with

    .

     And

     most

     importantly

    ,

     I

     am

     here

     to

     be

     your

     personal

     helper

     and

     guide

    .

     Your

     AI

     assistant

    ,

     [

    Name

    ].
    


    Tell

     me

     more

     about

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     for

     its

     iconic

     E

    iff

    el

     Tower

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     bustling

     streets

    .

     Paris

     is

     a

     city

     of

     culture

    ,

     art

    ,

     and

     architecture

    ,

     known

     for

     its

     French

     language

    ,

     cuisine

    ,

     and

     cuisine

    .

     The

     city

     is

     a

     popular

     tourist

     destination

     and

     is

     home

     to

     many

     museums

    ,

     landmarks

    ,

     and

     parks

    .

     Paris

     is

     also

     an

     international

     financial

     center

     and

     an

     important

     trade

     hub

     for

     France

     and

     Europe

    .

     The

     city

     is

     known

     for

     its

     cultural

     and

     artistic

     heritage

     and

     its

     role

     in

     French

     history

     and

     politics

    .

     With

     its

     blend

     of

     traditional

     French

     culture

     and

     modern

    ity

    ,

     Paris

     is

     a

     city

     that

     has

     capt

    ivated

     the

     world

     for

     centuries

    .

     It

     is

     the

     most

     visited

     city

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     increasingly

     complex

     and

     uncertain

    ,

     with

     many

     potential

     directions

     and

     possibilities

    .

     Here

     are

     some

     potential

     trends

     to

     consider

    :
    


    1

    .

     **

    Integration

     with

     human

     intelligence

    **:

     As

     AI

     becomes

     more

     sophisticated

    ,

     its

     integration

     with

     human

     intelligence

     is

     likely

     to

     become

     more

     seamless

    .

     AI

     systems

     could

     assist

    ,

     teach

    ,

     and

     even

     control

     human

     decision

    -making

     processes

    .
    


    2

    .

     **

    Art

    ificial

     general

     intelligence

     (

    AG

    I

    )**

    :

     While

     AG

    I

     is

     still

     a

     distant

     goal

    ,

     it

     is

     likely

     to

     become

     a

     reality

    .

     This

     involves

     the

     development

     of

     systems

     that

     can

     perform

     tasks

     that

     are

     considered

     human

    -level

    ,

     such

     as

     reasoning

    ,

     creativity

    ,

     and

     problem

    -solving

    .
    


    3

    .

     **

    De

    e

    pest

     Learning

    **:

     AI

     is

     increasingly

     using

    



```python
llm.shutdown()
```
