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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.95it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.95it/s]


    2026-04-28 04:12:12,279 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 04:12:12] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:22,  4.60s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.41it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.22it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  8.95it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.95it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=192):  52%|█████▏    | 30/58 [00:05<00:01, 14.84it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.60it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.60it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.60it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.60it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.60it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.60it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.60it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.60it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.60it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.60it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.60it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.40it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.40it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.40it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.40it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.40it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.40it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.40it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.96it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.51 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.48 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.48 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.48 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.48 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.48 GB):   3%|▎         | 2/58 [00:00<00:02, 18.72it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=56.48 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.47 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.46 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=4608 avail_mem=56.45 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.45 GB):   9%|▊         | 5/58 [00:00<00:02, 21.68it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.45 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.78it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.42 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.78it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.78it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.78it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=56.35 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.78it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.34 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.33 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.33 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.82 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.03it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.82 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.31it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.79 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.31it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=55.78 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.31it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.78 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.31it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=55.78 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.02it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.76 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.02it/s]Capturing num tokens (num_tokens=960 avail_mem=55.78 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.02it/s] Capturing num tokens (num_tokens=896 avail_mem=55.77 GB):  34%|███▍      | 20/58 [00:01<00:02, 16.02it/s]

    Capturing num tokens (num_tokens=896 avail_mem=55.77 GB):  40%|███▉      | 23/58 [00:01<00:02, 14.89it/s]Capturing num tokens (num_tokens=832 avail_mem=55.77 GB):  40%|███▉      | 23/58 [00:01<00:02, 14.89it/s]Capturing num tokens (num_tokens=768 avail_mem=55.77 GB):  40%|███▉      | 23/58 [00:01<00:02, 14.89it/s]Capturing num tokens (num_tokens=768 avail_mem=55.77 GB):  43%|████▎     | 25/58 [00:01<00:02, 14.42it/s]Capturing num tokens (num_tokens=704 avail_mem=55.76 GB):  43%|████▎     | 25/58 [00:01<00:02, 14.42it/s]

    Capturing num tokens (num_tokens=640 avail_mem=55.76 GB):  43%|████▎     | 25/58 [00:01<00:02, 14.42it/s]Capturing num tokens (num_tokens=640 avail_mem=55.76 GB):  47%|████▋     | 27/58 [00:01<00:02, 14.02it/s]Capturing num tokens (num_tokens=576 avail_mem=55.76 GB):  47%|████▋     | 27/58 [00:01<00:02, 14.02it/s]Capturing num tokens (num_tokens=512 avail_mem=55.74 GB):  47%|████▋     | 27/58 [00:01<00:02, 14.02it/s]

    Capturing num tokens (num_tokens=512 avail_mem=55.74 GB):  50%|█████     | 29/58 [00:01<00:02, 13.60it/s]Capturing num tokens (num_tokens=480 avail_mem=55.76 GB):  50%|█████     | 29/58 [00:01<00:02, 13.60it/s]Capturing num tokens (num_tokens=448 avail_mem=55.76 GB):  50%|█████     | 29/58 [00:01<00:02, 13.60it/s]Capturing num tokens (num_tokens=448 avail_mem=55.76 GB):  53%|█████▎    | 31/58 [00:01<00:02, 13.43it/s]Capturing num tokens (num_tokens=416 avail_mem=55.76 GB):  53%|█████▎    | 31/58 [00:01<00:02, 13.43it/s]

    Capturing num tokens (num_tokens=384 avail_mem=55.75 GB):  53%|█████▎    | 31/58 [00:01<00:02, 13.43it/s]Capturing num tokens (num_tokens=384 avail_mem=55.75 GB):  57%|█████▋    | 33/58 [00:02<00:01, 13.18it/s]Capturing num tokens (num_tokens=352 avail_mem=55.75 GB):  57%|█████▋    | 33/58 [00:02<00:01, 13.18it/s]Capturing num tokens (num_tokens=320 avail_mem=55.74 GB):  57%|█████▋    | 33/58 [00:02<00:01, 13.18it/s]

    Capturing num tokens (num_tokens=320 avail_mem=55.74 GB):  60%|██████    | 35/58 [00:02<00:01, 13.10it/s]Capturing num tokens (num_tokens=288 avail_mem=55.74 GB):  60%|██████    | 35/58 [00:02<00:01, 13.10it/s]Capturing num tokens (num_tokens=256 avail_mem=55.74 GB):  60%|██████    | 35/58 [00:02<00:01, 13.10it/s]Capturing num tokens (num_tokens=256 avail_mem=55.74 GB):  64%|██████▍   | 37/58 [00:02<00:01, 12.96it/s]Capturing num tokens (num_tokens=240 avail_mem=55.73 GB):  64%|██████▍   | 37/58 [00:02<00:01, 12.96it/s]

    Capturing num tokens (num_tokens=224 avail_mem=55.73 GB):  64%|██████▍   | 37/58 [00:02<00:01, 12.96it/s]Capturing num tokens (num_tokens=224 avail_mem=55.73 GB):  67%|██████▋   | 39/58 [00:02<00:01, 12.81it/s]Capturing num tokens (num_tokens=208 avail_mem=55.73 GB):  67%|██████▋   | 39/58 [00:02<00:01, 12.81it/s]Capturing num tokens (num_tokens=192 avail_mem=55.73 GB):  67%|██████▋   | 39/58 [00:02<00:01, 12.81it/s]

    Capturing num tokens (num_tokens=192 avail_mem=55.73 GB):  71%|███████   | 41/58 [00:02<00:01, 12.69it/s]Capturing num tokens (num_tokens=176 avail_mem=55.72 GB):  71%|███████   | 41/58 [00:02<00:01, 12.69it/s]Capturing num tokens (num_tokens=160 avail_mem=55.72 GB):  71%|███████   | 41/58 [00:02<00:01, 12.69it/s]Capturing num tokens (num_tokens=160 avail_mem=55.72 GB):  74%|███████▍  | 43/58 [00:02<00:01, 11.96it/s]Capturing num tokens (num_tokens=144 avail_mem=55.72 GB):  74%|███████▍  | 43/58 [00:02<00:01, 11.96it/s]

    Capturing num tokens (num_tokens=128 avail_mem=55.72 GB):  74%|███████▍  | 43/58 [00:02<00:01, 11.96it/s]Capturing num tokens (num_tokens=128 avail_mem=55.72 GB):  78%|███████▊  | 45/58 [00:03<00:01, 12.14it/s]Capturing num tokens (num_tokens=112 avail_mem=55.71 GB):  78%|███████▊  | 45/58 [00:03<00:01, 12.14it/s]Capturing num tokens (num_tokens=96 avail_mem=55.71 GB):  78%|███████▊  | 45/58 [00:03<00:01, 12.14it/s] Capturing num tokens (num_tokens=80 avail_mem=55.71 GB):  78%|███████▊  | 45/58 [00:03<00:01, 12.14it/s]

    Capturing num tokens (num_tokens=80 avail_mem=55.71 GB):  83%|████████▎ | 48/58 [00:03<00:00, 13.80it/s]Capturing num tokens (num_tokens=64 avail_mem=55.70 GB):  83%|████████▎ | 48/58 [00:03<00:00, 13.80it/s]Capturing num tokens (num_tokens=48 avail_mem=55.70 GB):  83%|████████▎ | 48/58 [00:03<00:00, 13.80it/s]Capturing num tokens (num_tokens=48 avail_mem=55.70 GB):  86%|████████▌ | 50/58 [00:03<00:00, 13.38it/s]Capturing num tokens (num_tokens=32 avail_mem=55.70 GB):  86%|████████▌ | 50/58 [00:03<00:00, 13.38it/s]

    Capturing num tokens (num_tokens=28 avail_mem=55.69 GB):  86%|████████▌ | 50/58 [00:03<00:00, 13.38it/s]Capturing num tokens (num_tokens=28 avail_mem=55.69 GB):  90%|████████▉ | 52/58 [00:03<00:00, 12.97it/s]Capturing num tokens (num_tokens=24 avail_mem=55.69 GB):  90%|████████▉ | 52/58 [00:03<00:00, 12.97it/s]Capturing num tokens (num_tokens=20 avail_mem=55.68 GB):  90%|████████▉ | 52/58 [00:03<00:00, 12.97it/s]

    Capturing num tokens (num_tokens=20 avail_mem=55.68 GB):  93%|█████████▎| 54/58 [00:03<00:00, 11.82it/s]Capturing num tokens (num_tokens=16 avail_mem=55.68 GB):  93%|█████████▎| 54/58 [00:03<00:00, 11.82it/s]Capturing num tokens (num_tokens=12 avail_mem=55.68 GB):  93%|█████████▎| 54/58 [00:03<00:00, 11.82it/s]Capturing num tokens (num_tokens=12 avail_mem=55.68 GB):  97%|█████████▋| 56/58 [00:03<00:00, 11.77it/s]Capturing num tokens (num_tokens=8 avail_mem=55.68 GB):  97%|█████████▋| 56/58 [00:03<00:00, 11.77it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=55.67 GB):  97%|█████████▋| 56/58 [00:03<00:00, 11.77it/s]Capturing num tokens (num_tokens=4 avail_mem=55.67 GB): 100%|██████████| 58/58 [00:04<00:00, 12.32it/s]Capturing num tokens (num_tokens=4 avail_mem=55.67 GB): 100%|██████████| 58/58 [00:04<00:00, 14.33it/s]


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
    Generated text:  Jim and I'm a big fan of the Harry Potter series. I thought I would tell you about a story that I read recently. The book was about a girl named Rose who was in a new school in a small town. She met a boy named Harry who was a bit of a fish out of water. They quickly became friends and were able to become good friends. Rose really enjoyed being in a new environment and having friends. She had some trouble with the school rules and her peers, but she was determined to prove herself to everyone she met. She read every book they had and did her homework. She had to understand the rules
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to reduce the greenhouse gas emissions from the cars that run the national infrastructure. The president wants to achieve this by incentivizing the use of electric cars. If the cost of electric cars is $20,000, and the president wants to reduce emissions by 30%, how much money in total does the president need to spend on the incentives for each of the cars? To determine how much money the president needs to spend on incentives for each electric car, we can follow these steps:
    
    1. **Identify the total cost of electric cars:**
       The cost of electric cars is $20,000
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris. The main building of the capital is the Eiffel Tower. It is a symbol of Paris. The Eiffel Tower is located in the center of the city. In the center of Paris there are several important buildings. The Eiffel Tower is the second one. The Eiffel Tower is one of the most famous buildings of the capital. It is about 300 meters above sea level. It is one of the tallest buildings in the world. The Eiffel Tower is the oldest building in Paris. It was built in 1889. It has 828 steps
    ===============================
    Prompt: The future of AI is
    Generated text:  here. With the adoption of AI technology becoming widespread, many people are beginning to think about how to use AI in their lives. AI has become a powerful tool that can improve productivity, reduce costs, and increase efficiency. But, before you can fully realize the benefits of AI, you need to understand the basics. In this article, we will explain what AI is, what it can do, and how it can help you.
    AI is a type of technology that uses algorithms and machine learning to automate processes and solve problems. It is a rapidly growing field, with applications in various industries, including finance, healthcare, retail, transportation, and


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for ways to [job title] and I'm always eager to learn new things. I'm a [job title] at [company name], and I'm always looking for ways to [job title] and I'm always eager to learn new things. I'm a [job title] at [company name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic hub with a rich history dating back to the Roman Empire and the Renaissance. It is a popular tourist destination and a major economic and political center in Europe. The city is also known for its cuisine, including French cuisine, and its fashion industry. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is also home to many international organizations and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased automation and robotics: As AI becomes more advanced, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare to transportation.
    
    2. AI-powered healthcare: AI is already being used in healthcare to diagnose and treat diseases, and we can expect to see even more advanced applications in the future, such as personalized medicine and virtual consultations.
    
    3. AI-powered education: AI is already being used in education to personalize learning experiences, and we can expect to see even more advanced applications in the future, such as adaptive learning platforms and personalized tutoring.
    
    4.
    


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
    Generated text:  [Name], and I am a [type of character, such as a superhero, alien, etc.]. I'm currently [current location or state], and I'm here to [mention a specific mission or task]. My [current profession or role], [Type of character], [Name], is here to [mention a specific objective or challenge].
    As an [type of character], I am here to [mention a specific goal or challenge], and I am ready to [mention any positive attributes or qualities]. Thank you for having me. 🌍✨
    Your answer should be concise and informative. Don't hesitate to add any additional
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city with the most iconic landmark, Notre-Dame Cathedral, and the most important symbol of French culture and history. Paris is the heart of the French Republic and is home to many renowned art, music, and theater venues. The city is also known as the "City of Light" and is a cultural hub that draws millions of visitors every year. Paris is often referred to as the "City of Love" due to its famous gardens and romantic architecture. With its rich history and diverse culture, Paris has been a major center of learning, art, and innovation for centuries. As of 2023, Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by significant advancements in areas such as machine learning, deep learning, and natural language processing. AI will likely become more capable of understanding and generating human-like language, leading to more sophisticated and intelligent virtual assistants and chatbots. This will also result in more sophisticated autonomous vehicles, drones, and robots, which can perform tasks such as self-driving, disaster response, and medical care.
    
    In addition, AI will likely continue to become more capable of performing tasks in fields such as manufacturing, healthcare, and finance, as it will become more integrated into these industries. This will lead to greater efficiency and productivity, as well as lower costs


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

    ]

     and

     I

    'm

     a

     [

    role

    ]

     in

     this

     story

    .

     What

     can

     you

     tell

     me

     about

     yourself

     and

     what

     brings

     you

     here

    ?

     What

     kind

     of

     story

     do

     you

     want

     to

     tell

     me

    ?

     As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     a

     physical

     presence

     and

     so

     I

     don

    't

     have

     a

     "

    self

    -int

    roduction

    "

     like

     a

     human

     does

    .

     However

    ,

     I

    'm

     always

     here

     to

     assist

     you

     with

     any

     questions

     or

     tasks

     you

     may

     have

    !

     Just

     let

     me

     know

     what

     you

    'd

     like

     me

     to

     do

     and

     I

    'll

     do

     my

     best

     to

     help

     you

    .

     What

     do

     you

     want

     to

     know

     about

     me

    ?

     I

    'm

     a

     [

    role

    ]

     in

     this

     story

    ,

     so

     feel

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     south

     of

     the

     country

     and

     known

     for

     its

     rich

     history

    ,

     vibrant

     culture

    ,

     and

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

     and

     the

     E

    iff

    el

     Tower

    .

     Paris

     is

     also

     renowned

     for

     its

     fashion

     industry

    ,

     art

     scene

    ,

     and

     its

     influence

     on

     global

     culture

    .

     It

     is

     the

     third

    -largest

     city

     in

     Europe

     by

     population

     and

     one

     of

     the

     world

    's

     most

     popular

     tourist

     destinations

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     promising

    ,

     with

     many

     possibilities

     and

     potential

     breakthrough

    s

     on

     the

     horizon

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

     Automation

     and

     Efficiency

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     even

     more

     automation

     in

     various

     industries

    ,

     leading

     to

     increased

     efficiency

     and

     productivity

    .

     This

     could

     mean

     that

     we

     see

     more

     autonomous

     vehicles

    ,

     robots

    ,

     and

     other

     automation

     in

     everyday

     life

    .
    


    2

    .

     AI

     In

    clus

    ivity

     and

     Equality

    :

     With

     the

     growth

     of

     AI

    ,

     we

     can

     expect

     to

     see

     greater

     emphasis

     on

     ensuring

     that

     AI

     is

     accessible

     and

     inclusive

    ,

     meaning

     that

     it

     is

     designed

     to

     work

     for

     all

     people

    ,

     regardless

     of

     their

     abilities

     or

     backgrounds

    .
    


    3

    .

     AI

     Ethics

    



```python
llm.shutdown()
```
