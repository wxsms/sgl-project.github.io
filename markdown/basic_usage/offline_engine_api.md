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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.96it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.94it/s]


    2026-04-09 18:22:20,098 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 18:22:20] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:37,  2.77s/it]Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:02<01:07,  1.20s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:02<01:07,  1.20s/it]Compiling num tokens (num_tokens=6656):   3%|▎         | 2/58 [00:02<01:07,  1.20s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:25,  2.08it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:25,  2.08it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:03<00:25,  2.08it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:25,  2.08it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:03<00:11,  4.37it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:03<00:11,  4.37it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:03<00:11,  4.37it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:03<00:11,  4.37it/s]

    Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:03<00:11,  4.37it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:05,  7.96it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:05,  7.96it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:05,  7.96it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:05,  7.96it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:05,  7.96it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:05,  7.96it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:05,  7.96it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:05,  7.96it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:05,  7.96it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:02, 17.12it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:02, 17.12it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:02, 17.12it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:02, 17.12it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:02, 17.12it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 17.12it/s]

    Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:02, 17.12it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:02, 17.12it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 24.89it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 24.89it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 24.89it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 24.89it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 24.89it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 24.89it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 24.89it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 24.89it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 32.49it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 32.49it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 32.49it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 32.49it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 32.49it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 32.49it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 32.49it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 37.22it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 37.22it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 37.22it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 37.22it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 37.22it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 37.22it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 37.22it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 40.30it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 40.30it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 41.90it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 41.90it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 41.90it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 41.90it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 41.90it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 41.90it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 41.90it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 41.90it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 48.46it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 14.45it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.77 GB):   3%|▎         | 2/58 [00:00<00:05, 10.49it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.77 GB):   3%|▎         | 2/58 [00:00<00:05, 10.49it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=118.76 GB):   3%|▎         | 2/58 [00:00<00:05, 10.49it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   3%|▎         | 2/58 [00:00<00:05, 10.49it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.77 GB):   9%|▊         | 5/58 [00:00<00:03, 16.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:03, 16.67it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:03, 16.67it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:03, 16.67it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):   9%|▊         | 5/58 [00:00<00:03, 16.67it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=118.76 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.75 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.59 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.58 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.58 GB):  16%|█▌        | 9/58 [00:00<00:02, 22.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.58 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.04it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.04it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.57 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.04it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.34 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.04it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=118.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.27it/s]Capturing num tokens (num_tokens=1792 avail_mem=118.20 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=118.20 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=118.01 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.27it/s]Capturing num tokens (num_tokens=1024 avail_mem=117.99 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.27it/s]Capturing num tokens (num_tokens=1024 avail_mem=117.99 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=960 avail_mem=118.01 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.93it/s] Capturing num tokens (num_tokens=896 avail_mem=118.01 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.93it/s]

    Capturing num tokens (num_tokens=832 avail_mem=118.00 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=768 avail_mem=118.00 GB):  36%|███▌      | 21/58 [00:00<00:01, 27.93it/s]Capturing num tokens (num_tokens=768 avail_mem=118.00 GB):  43%|████▎     | 25/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=704 avail_mem=118.00 GB):  43%|████▎     | 25/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=640 avail_mem=117.99 GB):  43%|████▎     | 25/58 [00:00<00:01, 29.44it/s]Capturing num tokens (num_tokens=576 avail_mem=117.99 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.44it/s]Capturing num tokens (num_tokens=512 avail_mem=117.98 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.44it/s]

    Capturing num tokens (num_tokens=512 avail_mem=117.98 GB):  50%|█████     | 29/58 [00:01<00:01, 20.30it/s]Capturing num tokens (num_tokens=480 avail_mem=118.00 GB):  50%|█████     | 29/58 [00:01<00:01, 20.30it/s]Capturing num tokens (num_tokens=448 avail_mem=117.99 GB):  50%|█████     | 29/58 [00:01<00:01, 20.30it/s]Capturing num tokens (num_tokens=416 avail_mem=117.99 GB):  50%|█████     | 29/58 [00:01<00:01, 20.30it/s]Capturing num tokens (num_tokens=384 avail_mem=117.99 GB):  50%|█████     | 29/58 [00:01<00:01, 20.30it/s]Capturing num tokens (num_tokens=352 avail_mem=117.98 GB):  50%|█████     | 29/58 [00:01<00:01, 20.30it/s]Capturing num tokens (num_tokens=352 avail_mem=117.98 GB):  59%|█████▊    | 34/58 [00:01<00:00, 24.67it/s]Capturing num tokens (num_tokens=320 avail_mem=117.98 GB):  59%|█████▊    | 34/58 [00:01<00:00, 24.67it/s]Capturing num tokens (num_tokens=288 avail_mem=117.98 GB):  59%|█████▊    | 34/58 [00:01<00:00, 24.67it/s]Capturing num tokens (num_tokens=256 avail_mem=117.97 GB):  59%|█████▊    | 34/58 [00:01<00:00, 24.67it/s]

    Capturing num tokens (num_tokens=256 avail_mem=117.97 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.67it/s]Capturing num tokens (num_tokens=240 avail_mem=117.97 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.67it/s]

    Capturing num tokens (num_tokens=224 avail_mem=118.84 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.67it/s]Capturing num tokens (num_tokens=208 avail_mem=137.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 25.67it/s]Capturing num tokens (num_tokens=208 avail_mem=137.27 GB):  69%|██████▉   | 40/58 [00:01<00:01, 14.84it/s]

    Capturing num tokens (num_tokens=192 avail_mem=137.27 GB):  69%|██████▉   | 40/58 [00:01<00:01, 14.84it/s]Capturing num tokens (num_tokens=176 avail_mem=137.27 GB):  69%|██████▉   | 40/58 [00:02<00:01, 14.84it/s]Capturing num tokens (num_tokens=160 avail_mem=137.26 GB):  69%|██████▉   | 40/58 [00:02<00:01, 14.84it/s]Capturing num tokens (num_tokens=160 avail_mem=137.26 GB):  74%|███████▍  | 43/58 [00:02<00:00, 15.09it/s]Capturing num tokens (num_tokens=144 avail_mem=137.26 GB):  74%|███████▍  | 43/58 [00:02<00:00, 15.09it/s]

    Capturing num tokens (num_tokens=128 avail_mem=137.26 GB):  74%|███████▍  | 43/58 [00:02<00:00, 15.09it/s]Capturing num tokens (num_tokens=112 avail_mem=137.24 GB):  74%|███████▍  | 43/58 [00:02<00:00, 15.09it/s]Capturing num tokens (num_tokens=112 avail_mem=137.24 GB):  79%|███████▉  | 46/58 [00:02<00:00, 15.19it/s]Capturing num tokens (num_tokens=96 avail_mem=136.76 GB):  79%|███████▉  | 46/58 [00:02<00:00, 15.19it/s] Capturing num tokens (num_tokens=80 avail_mem=136.75 GB):  79%|███████▉  | 46/58 [00:02<00:00, 15.19it/s]Capturing num tokens (num_tokens=64 avail_mem=136.62 GB):  79%|███████▉  | 46/58 [00:02<00:00, 15.19it/s]

    Capturing num tokens (num_tokens=48 avail_mem=136.59 GB):  79%|███████▉  | 46/58 [00:02<00:00, 15.19it/s]Capturing num tokens (num_tokens=32 avail_mem=136.58 GB):  79%|███████▉  | 46/58 [00:02<00:00, 15.19it/s]Capturing num tokens (num_tokens=32 avail_mem=136.58 GB):  88%|████████▊ | 51/58 [00:02<00:00, 20.35it/s]Capturing num tokens (num_tokens=28 avail_mem=136.58 GB):  88%|████████▊ | 51/58 [00:02<00:00, 20.35it/s]Capturing num tokens (num_tokens=24 avail_mem=136.58 GB):  88%|████████▊ | 51/58 [00:02<00:00, 20.35it/s]Capturing num tokens (num_tokens=20 avail_mem=136.57 GB):  88%|████████▊ | 51/58 [00:02<00:00, 20.35it/s]Capturing num tokens (num_tokens=16 avail_mem=136.57 GB):  88%|████████▊ | 51/58 [00:02<00:00, 20.35it/s]Capturing num tokens (num_tokens=12 avail_mem=136.57 GB):  88%|████████▊ | 51/58 [00:02<00:00, 20.35it/s]Capturing num tokens (num_tokens=12 avail_mem=136.57 GB):  97%|█████████▋| 56/58 [00:02<00:00, 25.30it/s]Capturing num tokens (num_tokens=8 avail_mem=136.56 GB):  97%|█████████▋| 56/58 [00:02<00:00, 25.30it/s] Capturing num tokens (num_tokens=4 avail_mem=136.56 GB):  97%|█████████▋| 56/58 [00:02<00:00, 25.30it/s]

    Capturing num tokens (num_tokens=4 avail_mem=136.56 GB): 100%|██████████| 58/58 [00:02<00:00, 22.18it/s]


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
    Generated text:  John and I'm a software developer. I have some questions about computers and technology.
    
    1. What is a computer?
    
    2. How do I know if my computer is old or new?
    
    3. Is it possible to create a computer that is one hundred percent new?
    
    4. Can I use my old computer for a new project?
    
    5. Is it possible to buy a new computer if I have an old one?
    
    6. What are some ways to improve my computer?
    
    7. Is there anything wrong with my computer that I should not know about?
    
    8. How do I know if my computer is infected with malware or other viruses?
    
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  a presidential candidate. What is the most likely function of the ______ in this sentence?
    A. Explanatory function
    B. Experiential function
    C. Educational function
    D. Organizational function
    Answer: D
    
    Which of the following statements about the primary functions of a protocol is incorrect? ____ A. A protocol is a set of rules that ensure communication between applications or devices, where the protocol is defined in a specific language B. Protocols are not bound to a specific physical medium, and can be transmitted over the Internet C. Protocols define a set of rules to ensure communication between two or more devices, and
    ===============================
    Prompt: The capital of France is
    Generated text:  _________. [ ]
    A. Paris
    B. Moscow
    C. Berlin
    D. Tokyo
    Answer: A
    
    In the Great War, the most famous French writer was ____. [ ]
    A. Honoré de Balzac
    B. Victor Hugo
    C. Maxim Gorky
    D. Ernest Hemingway
    Answer: B
    
    China is the only major maritime power on the global stage. [ ]
    A. Correct
    B. Incorrect
    Answer: A
    
    "According to the World Health Organization, the incidence of parasitic diseases in China is low, especially in the south. In fact, the northwest region has
    ===============================
    Prompt: The future of AI is
    Generated text:  hard to predict. But what we do know is that it has the potential to change how we live our lives in many ways. In this article, we will explore the future of AI in a few key areas and what we can expect to see in the future.
    AI is the field of computer science that deals with the design and development of intelligent machines. These machines are able to learn from data and use it to make decisions and perform tasks. Some examples of AI include natural language processing, computer vision, and machine learning.
    The future of AI is likely to involve many more applications. It is possible that AI will be used to improve healthcare


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [hobby or activity], and I'm always looking for new ways to explore and discover new things. What's your favorite book or movie? I love [book/movie], and I'm always looking for new
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant nightlife and is a popular tourist destination. The city is home to many international institutions and organizations, including the French Academy of Sciences and the European Parliament. It is a major transportation hub and is a major economic center in Europe. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. The city is also known for its diverse cuisine and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems that can better understand and respond to human needs.
    
    2. Enhanced machine learning capabilities: AI is likely to become even more powerful and capable, with the ability to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective AI systems that can handle a wider range of tasks.
    
    3. Greater reliance on AI for decision-making: AI is likely
    


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
    Generated text:  [Name], and I'm a [Age] year old [Occupation] [Job Title]. I have a passion for [describe a hobby or interest that you enjoy doing]. I enjoy [mention something about your hobbies or interests]. I also love [describe a skill you have that can be used for [describe a specific purpose or hobby you have]} and [describe something that you have that you're proud of]. I'm a [describe your unique selling point or talent]. I'm [describe your personality trait or soft spot for a particular person]. Overall, I’m [describe your overall personality and personality type]. I’m excited to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the largest city in France and one of the most important cities in the world.
    
    Paris is the capital of France, located in the southern part of the country, on the French Riviera coast. The city has a rich history dating back to the Middle Ages, including the Notre-Dame Cathedral, which was founded in 1163 and remains one of the most important landmarks in Paris. Other notable buildings in Paris include the Eiffel Tower, the Louvre Museum, and the Musée de l'Orangerie. The city is known for its vibrant cultural scene, including the annual May Day celebrations,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve many different trends that will shape how we use and interact with AI technology. Here are some potential areas of development:
    
    1. Increased AI Transparency: With the rise of AI systems, there will be a greater emphasis on transparency and explainability. As AI systems become more sophisticated, we'll need to be able to understand how they are making decisions and predictions.
    
    2. Ethical AI: As AI systems become more sophisticated, there will be a growing focus on ethical considerations. This includes issues such as bias, accountability, and privacy.
    
    3. Interdisciplinary Applications: AI is currently being used in many different fields, but there will


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

    ],

     and

     I

     am

     a

     [

    Type

     of

     Character

    ]

     [

    Your

     Type

     of

     Character

    ].

     I

     am

     a

     [

    Your

     Type

     of

     Character

    ]

     in

     my

     [

    Your

     Profession

     or

     Area

     of

     Expert

    ise

    ].

     I

    've

     been

     [

    Your

     Starting

     Occupation

     or

     Career

    ],

     and

     I

    've

     always

     been

     [

    Your

     Personality

    ].

     My

     [

    Your

     Profession

     or

     Area

     of

     Expert

    ise

    ]

     is

    ...

     well

    ,

     whatever

     that

     is

    .

     I

    'm

     [

    Your

     Type

     of

     Character

    ]

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    Your

     Personality

    ]

     and

     [

    Your

     Career

     Goals

    ].

     So

    ,

     what

     brings

     you

     to

     [

    Your

     Location

    ]

     today

    ?

     This

     is

     [

    Your

     Location

    ]

     and

     [

    Your

     Name

    ],

     your

     fellow

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     historical

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

     also

     has

     many

     world

    -ren

    owned

     museums

     and

     cultural

     institutions

    ,

     including

     the

     Lou

    vre

     Museum

     and

     Mus

    ée

     d

    '

    Or

    say

    .

     Paris

     is

     a

     vibrant

     and

     diverse

     city

     with

     a

     rich

     cultural

     heritage

    ,

     known

     for

     its

     culinary

     traditions

    ,

     fashion

     industry

    ,

     and

     outdoor

     sports

     facilities

    .

     It

     is

     a

     popular

     tourist

     destination

    ,

     known

     for

     its

     beautiful

     architecture

    ,

     charming

     neighborhoods

    ,

     and

     annual

     festivals

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

     culture

    ,

     making

     it

     a

     must

    -

    visit

     destination

     for

     anyone

     interested

     in

     French

     culture

     and

     history

    .

     
    


    Therefore

    ,

     the

     concise

     factual

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     varied

    ,

     with

     many

     different

     possibilities

     depending

     on

     how

     we

     choose

     to

     use

     and

     develop

     the

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

     Improved

     accuracy

     and

     reliability

    :

     As

     AI

     technology

     continues

     to

     advance

    ,

     there

     will

     likely

     be

     further

     improvements

     in

     accuracy

     and

     reliability

    .

     This

     will

     require

     more

     data

     and

     more

     sophisticated

     algorithms

     to

     process

     and

     interpret

     the

     data

    .
    


    2

    .

     Increased

     collaboration

     between

     AI

     and

     human

     experts

    :

     As

     AI

     continues

     to

     become

     more

     sophisticated

    ,

     there

     will

     likely

     be

     increased

     collaboration

     between

     AI

     and

     human

     experts

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     law

    .
    


    3

    .

     Increased

     use

     of

     AI

     in

     the

     healthcare

     industry

    :

     AI

    -powered

     medical

     devices

     and

     software

     could

    



```python
llm.shutdown()
```
