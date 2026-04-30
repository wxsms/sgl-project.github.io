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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.46it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.45it/s]


    2026-04-30 18:52:43,527 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-30 18:52:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:33,  4.79s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:33,  4.79s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:33,  4.79s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:33,  4.79s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:33,  4.79s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:39,  1.36it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.07it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.07it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.16it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.16it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.16it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.16it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.16it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.16it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.16it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.16it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.16it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.16it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 14.80it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.45it/s]

    Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.45it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.45it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 27.77it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 27.77it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 33.45it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 33.45it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.34it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.09 GB):   2%|▏         | 1/58 [00:00<00:05,  9.65it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.06 GB):   2%|▏         | 1/58 [00:00<00:05,  9.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.05 GB):   2%|▏         | 1/58 [00:00<00:05,  9.65it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=72.05 GB):   5%|▌         | 3/58 [00:00<00:04, 11.98it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.05 GB):   5%|▌         | 3/58 [00:00<00:04, 11.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.05 GB):   5%|▌         | 3/58 [00:00<00:04, 11.98it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.05 GB):   9%|▊         | 5/58 [00:00<00:03, 14.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.04 GB):   9%|▊         | 5/58 [00:00<00:03, 14.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.03 GB):   9%|▊         | 5/58 [00:00<00:03, 14.33it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=72.03 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.03 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.03 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.02 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.02 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.97it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.02 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.02 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.01 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.01 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.97it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.01 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.01 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.00 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.18it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.00 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.18it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.00 GB):  24%|██▍       | 14/58 [00:00<00:01, 25.18it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.00 GB):  31%|███       | 18/58 [00:00<00:01, 29.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.99 GB):  31%|███       | 18/58 [00:00<00:01, 29.38it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.99 GB):  31%|███       | 18/58 [00:00<00:01, 29.38it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.97 GB):  31%|███       | 18/58 [00:00<00:01, 29.38it/s]Capturing num tokens (num_tokens=960 avail_mem=71.99 GB):  31%|███       | 18/58 [00:00<00:01, 29.38it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=71.99 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.79it/s]Capturing num tokens (num_tokens=896 avail_mem=71.67 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.79it/s]Capturing num tokens (num_tokens=832 avail_mem=71.95 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.79it/s]Capturing num tokens (num_tokens=768 avail_mem=71.94 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.79it/s]Capturing num tokens (num_tokens=704 avail_mem=71.94 GB):  38%|███▊      | 22/58 [00:01<00:01, 31.79it/s]Capturing num tokens (num_tokens=704 avail_mem=71.94 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.07it/s]Capturing num tokens (num_tokens=640 avail_mem=71.71 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.07it/s]

    Capturing num tokens (num_tokens=576 avail_mem=71.72 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.07it/s]Capturing num tokens (num_tokens=512 avail_mem=71.91 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.07it/s]Capturing num tokens (num_tokens=512 avail_mem=71.91 GB):  50%|█████     | 29/58 [00:01<00:01, 26.82it/s]Capturing num tokens (num_tokens=480 avail_mem=71.92 GB):  50%|█████     | 29/58 [00:01<00:01, 26.82it/s]Capturing num tokens (num_tokens=448 avail_mem=71.91 GB):  50%|█████     | 29/58 [00:01<00:01, 26.82it/s]Capturing num tokens (num_tokens=416 avail_mem=71.88 GB):  50%|█████     | 29/58 [00:01<00:01, 26.82it/s]

    Capturing num tokens (num_tokens=416 avail_mem=71.88 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.37it/s]Capturing num tokens (num_tokens=384 avail_mem=71.87 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.37it/s]Capturing num tokens (num_tokens=352 avail_mem=71.62 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.37it/s]Capturing num tokens (num_tokens=320 avail_mem=71.84 GB):  55%|█████▌    | 32/58 [00:01<00:01, 23.37it/s]Capturing num tokens (num_tokens=320 avail_mem=71.84 GB):  60%|██████    | 35/58 [00:01<00:01, 20.86it/s]Capturing num tokens (num_tokens=288 avail_mem=71.84 GB):  60%|██████    | 35/58 [00:01<00:01, 20.86it/s]

    Capturing num tokens (num_tokens=256 avail_mem=71.63 GB):  60%|██████    | 35/58 [00:01<00:01, 20.86it/s]Capturing num tokens (num_tokens=240 avail_mem=71.78 GB):  60%|██████    | 35/58 [00:01<00:01, 20.86it/s]Capturing num tokens (num_tokens=240 avail_mem=71.78 GB):  66%|██████▌   | 38/58 [00:01<00:00, 20.59it/s]Capturing num tokens (num_tokens=224 avail_mem=71.65 GB):  66%|██████▌   | 38/58 [00:01<00:00, 20.59it/s]Capturing num tokens (num_tokens=208 avail_mem=71.77 GB):  66%|██████▌   | 38/58 [00:01<00:00, 20.59it/s]Capturing num tokens (num_tokens=192 avail_mem=71.79 GB):  66%|██████▌   | 38/58 [00:01<00:00, 20.59it/s]

    Capturing num tokens (num_tokens=192 avail_mem=71.79 GB):  71%|███████   | 41/58 [00:01<00:00, 21.00it/s]Capturing num tokens (num_tokens=176 avail_mem=71.78 GB):  71%|███████   | 41/58 [00:01<00:00, 21.00it/s]Capturing num tokens (num_tokens=160 avail_mem=71.78 GB):  71%|███████   | 41/58 [00:01<00:00, 21.00it/s]Capturing num tokens (num_tokens=144 avail_mem=71.75 GB):  71%|███████   | 41/58 [00:01<00:00, 21.00it/s]Capturing num tokens (num_tokens=144 avail_mem=71.75 GB):  76%|███████▌  | 44/58 [00:01<00:00, 21.46it/s]Capturing num tokens (num_tokens=128 avail_mem=71.76 GB):  76%|███████▌  | 44/58 [00:01<00:00, 21.46it/s]Capturing num tokens (num_tokens=112 avail_mem=71.75 GB):  76%|███████▌  | 44/58 [00:02<00:00, 21.46it/s]

    Capturing num tokens (num_tokens=96 avail_mem=71.74 GB):  76%|███████▌  | 44/58 [00:02<00:00, 21.46it/s] Capturing num tokens (num_tokens=96 avail_mem=71.74 GB):  81%|████████  | 47/58 [00:02<00:00, 22.16it/s]Capturing num tokens (num_tokens=80 avail_mem=71.73 GB):  81%|████████  | 47/58 [00:02<00:00, 22.16it/s]Capturing num tokens (num_tokens=64 avail_mem=71.72 GB):  81%|████████  | 47/58 [00:02<00:00, 22.16it/s]Capturing num tokens (num_tokens=48 avail_mem=71.71 GB):  81%|████████  | 47/58 [00:02<00:00, 22.16it/s]Capturing num tokens (num_tokens=48 avail_mem=71.71 GB):  86%|████████▌ | 50/58 [00:02<00:00, 23.48it/s]Capturing num tokens (num_tokens=32 avail_mem=71.69 GB):  86%|████████▌ | 50/58 [00:02<00:00, 23.48it/s]Capturing num tokens (num_tokens=28 avail_mem=71.70 GB):  86%|████████▌ | 50/58 [00:02<00:00, 23.48it/s]

    Capturing num tokens (num_tokens=24 avail_mem=71.69 GB):  86%|████████▌ | 50/58 [00:02<00:00, 23.48it/s]Capturing num tokens (num_tokens=24 avail_mem=71.69 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.30it/s]Capturing num tokens (num_tokens=20 avail_mem=71.67 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.30it/s]Capturing num tokens (num_tokens=16 avail_mem=71.68 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.30it/s]Capturing num tokens (num_tokens=12 avail_mem=71.67 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.30it/s]Capturing num tokens (num_tokens=8 avail_mem=71.66 GB):  91%|█████████▏| 53/58 [00:02<00:00, 24.30it/s] Capturing num tokens (num_tokens=8 avail_mem=71.66 GB):  98%|█████████▊| 57/58 [00:02<00:00, 26.64it/s]Capturing num tokens (num_tokens=4 avail_mem=71.65 GB):  98%|█████████▊| 57/58 [00:02<00:00, 26.64it/s]Capturing num tokens (num_tokens=4 avail_mem=71.65 GB): 100%|██████████| 58/58 [00:02<00:00, 23.32it/s]


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
    Generated text:  Cathy.
    
    I am a PhD student at the Institute for Brain Research at the University of California, Berkeley. I research how changes in the brain affect our understanding of how our environment affects our behavior. My research focuses on the role of the amygdala, a region in the brain that regulates emotional responses and plays a role in stress and anxiety. I am also interested in how the amygdala changes over time, and how this change impacts our understanding of the brain’s ability to predict and respond to future events.
    
    I am interested in how we use the amygdala as a tool for navigation in our environment, and how changes in the
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 3 inches tall. If a person's height is a whole number of feet and a person's height is greater than 5 feet, then what is the minimum possible height?
    
    To determine the minimum possible height of the president of the United States, we need to consider the given constraints: the president's height is 5 feet 3 inches and it must be a whole number of feet and greater than 5 feet.
    
    First, let's convert the height from inches to feet. Since there are 12 inches in a foot, we can express the height in feet as:
    \[ 5 \text{ feet }
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. New York City
    D. Moscow
    A. Paris
    
    Paris is the capital of France and is the largest city in the country. It is located in the northwestern part of France and is known for its beautiful architecture, museums, and vibrant cultural scene. The city is also home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. New York City is also the capital of the United States, while Moscow is the capital of Russia. London, on the other hand, is the capital of the United Kingdom and is known for
    ===============================
    Prompt: The future of AI is
    Generated text:  fundamentally about how we will interact with technology. Artificial intelligence is changing the way we work, communicate, and even interact with each other. However, as AI technology continues to evolve, it is essential to understand the impact on our daily lives and the ethical implications of the technology we use.
    One of the most significant aspects of AI technology is its ability to enable rapid and cost-effective innovation. AI has the potential to accelerate the pace of development and bring new solutions to a wide range of problems. This has led to the development of a wide range of applications and technologies, such as machine learning, natural language processing, and computer vision.
    However,


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working here for [number of years] years. I am passionate about [reason for being at the company]. I enjoy [reason for being at the company]. I am always looking for ways to [reason for being at the company]. I am a [reason for being at the company]. I am a [reason for being at the company]. I am a [reason for being at the company]. I am a [reason for being at the company]. I am a [reason for being at the company]. I am a [reason for being
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the third-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also famous for its rich history, including the French Revolution and the French Revolution Museum. Paris is a cultural and economic center of France and a major tourist destination. It is home to many famous French artists, writers, and musicians. The city is also known for its fashion industry, with many famous fashion designers and boutiques. Paris is a vibrant and diverse city with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased automation: AI will continue to automate many tasks, from manufacturing to customer service. This will lead to increased efficiency and productivity, but it will also create new jobs and raise ethical concerns.
    
    2. AI will become more integrated with other technologies: AI will become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This will allow AI to learn from and adapt to new data, leading to more accurate and personalized predictions.
    
    3. AI will become more ethical: As AI becomes more integrated with other technologies, there will be a growing emphasis on ethical
    


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
    Generated text:  [insert name]. I am [insert your profession or identity]. I bring a fresh perspective to [insert your field or field of work], and I love helping people. I'm confident, reliable, and a great team player. I'm [insert any relevant achievements or accomplishments]. I'm always looking for new opportunities and I'm eager to learn and grow. How can I get to know you better? Are you interested in learning more about me? I'm happy to answer any questions you might have about me. Please feel free to ask me anything! [Insert name] [insert name] [Insert name] [insert name] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital of France and serves as the cultural, political, and economic center of the country. The city is known for its romantic architecture, world-famous museums, and iconic landmarks like the Eiffel Tower and the Louvre Museum. Its climate is mild year-round, making it a popular tourist destination. The city is also a hub for industry, finance, and media, attracting many international businesses and events. Paris is a cultural center that has played a significant role in shaping French history and culture. As of 2021, the population of Paris is approximately 2.2 million. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are some potential trends that could shape the industry:
    
    1. Augmented Intelligence: As AI technology advances, we may see a shift towards augmented intelligence, where AI is integrated into our daily lives. This could include virtual assistants like Siri, Alexa, and Google Assistant, as well as more advanced AI systems that can communicate and interact with humans.
    
    2. Machine Learning and Deep Learning: AI will continue to evolve, with new algorithms and models emerging. Machine learning and deep learning will become increasingly important, allowing AI systems to learn from data and improve their performance over time.
    
    3. Artificial General Intelligence: This is the ultimate


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

    .

     I

    'm

     __

    ________

    .

     I

    'm

     a

    /an

     __

    ________

    ,

     and

     I

    'm

     here

     to

     __

    ________

    .

     I

     hope

     I

     can

     help

     you

     with

     your

     __

    ________

    .
    


    And

     welcome

     to

     the

     world

     of

     __

    ________

    .

     I

    'm

     so

     glad

     I

     get

     to

     know

     you

    !

     What

     is

     your

     name

    ,

     what

     are

     you

     currently

     doing

     here

    ,

     and

     what

     are

     you

     here

     to

     do

    ?

     Please

     do

     tell

    !

     
    


    ---

     
    


    These

     are

     the

     elements

     of

     a

     self

    -int

    roduction

    ,

     but

     you

     can

     adapt

     them

     to

     the

     specific

     needs

     and

     environment

     of

     your

     character

    .

     The

     key

     is

     to

     be

     honest

    ,

     friendly

    ,

     and

     to

     the

     point

    ,

     so

     that

     your

     character

     feels

     at

     ease

     and

     comfortable

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     bustling

     city

     located

     in

     the

     north

    western

     region

     of

     the

     country

    .

     It

     is

     the

     cultural

     and

     economic

     center

     of

     the

     country

    ,

     and

     is

     known

     for

     its

     beautiful

     architecture

    ,

     French

     cuisine

    ,

     and

     world

    -class

     museums

     and

     attractions

    .

     The

     city

     is

     also

     the

     headquarters

     of

     many

     French

     government

     offices

     and

     is

     a

     popular

     tourist

     destination

    .

     Its

     annual

     tourism

     industry

     is

     one

     of

     the

     largest

     in

     the

     world

    ,

     with

     millions

     of

     visitors

     annually

     visiting

     the

     city

     to

     see

     its

     landmarks

     and

     experience

     its

     culture

    .

     Paris

     is

     known

     for

     its

     romantic

     ambiance

    ,

     cultural

     richness

    ,

     and

     iconic

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

     Palace

     of

     Vers

    ailles

    .

     Its

     long

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     increasingly

     influenced

     by

     several

     key

     trends

    ,

     including

    :
    


    1

    .

     Increased

     complexity

    :

     AI

     systems

     are

     becoming

     more

     complex

    ,

     with

     the

     ability

     to

     learn

     and

     adapt

     as

     they

     encounter

     new

     data

    .
    


    2

    .

     Increased

     data

     volume

     and

     diversity

    :

     AI

     systems

     need

     more

     data

     to

     learn

     effectively

    ,

     and

     the

     volume

     and

     diversity

     of

     data

     available

     will

     continue

     to

     grow

    .
    


    3

    .

     AI

     cognitive

     scalability

    :

     AI

     systems

     will

     continue

     to

     improve

     in

     terms

     of

     their

     ability

     to

     process

     more

     data

    ,

     while

     also

     increasing

     their

     ability

     to

     adapt

     to

     new

     situations

    .
    


    4

    .

     AI

     ethical

     considerations

    :

     As

     AI

     systems

     become

     more

     complex

    ,

     there

     will

     be

     greater

     concern

     about

     their

     impact

     on

     society

     and

     the

     environment

    .
    


    5

    .

     AI

     collaboration

    



```python
llm.shutdown()
```
