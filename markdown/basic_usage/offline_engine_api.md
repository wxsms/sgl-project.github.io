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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.81it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.80it/s]


    2026-05-18 01:39:54,588 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-18 01:39:54] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:23,  4.63s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.79it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:05,  6.88it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:05,  6.88it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:05,  6.88it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:05,  6.88it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:05,  6.88it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:05<00:05,  6.88it/s] 

    Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:05<00:05,  6.88it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:03, 10.67it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 17.05it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 17.05it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 17.05it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 17.05it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 17.05it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 17.05it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 17.05it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 17.05it/s]

    Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 17.05it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:05<00:00, 24.39it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:05<00:00, 24.39it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:05<00:00, 24.39it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:05<00:00, 24.39it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:05<00:00, 24.39it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 24.39it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 24.39it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 24.39it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 30.71it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 30.71it/s]

    Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 30.71it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 36.81it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 36.81it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 36.81it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 36.81it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 36.81it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 36.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:03, 14.23it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 14.23it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:03, 14.23it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   7%|▋         | 4/58 [00:00<00:03, 13.86it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   7%|▋         | 4/58 [00:00<00:03, 13.86it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   7%|▋         | 4/58 [00:00<00:03, 13.86it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):  10%|█         | 6/58 [00:00<00:03, 14.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):  10%|█         | 6/58 [00:00<00:03, 14.26it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):  10%|█         | 6/58 [00:00<00:03, 14.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  10%|█         | 6/58 [00:00<00:03, 14.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.98it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.98it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:03, 15.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  21%|██        | 12/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  21%|██        | 12/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.57 GB):  21%|██        | 12/58 [00:00<00:02, 18.79it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  21%|██        | 12/58 [00:00<00:02, 18.79it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.54it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.54it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.54it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.56 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  29%|██▉       | 17/58 [00:00<00:01, 25.54it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.54 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.32it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  36%|███▌      | 21/58 [00:00<00:01, 28.32it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.32it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  36%|███▌      | 21/58 [00:01<00:01, 28.32it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:01<00:02, 16.17it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:01<00:02, 16.17it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:01<00:02, 16.17it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:01<00:02, 16.17it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:01<00:02, 16.17it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:01<00:02, 16.17it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:01<00:01, 21.41it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:01<00:01, 21.41it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:01<00:01, 21.41it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:01<00:01, 21.41it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:01<00:01, 21.41it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.41it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.41it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.41it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:01, 23.41it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  62%|██████▏   | 36/58 [00:01<00:00, 23.45it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  62%|██████▏   | 36/58 [00:01<00:00, 23.45it/s]

    Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  62%|██████▏   | 36/58 [00:01<00:00, 23.45it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  62%|██████▏   | 36/58 [00:01<00:00, 23.45it/s]Capturing num tokens (num_tokens=224 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.03it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.03it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.03it/s]Capturing num tokens (num_tokens=176 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.03it/s]Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  67%|██████▋   | 39/58 [00:01<00:00, 24.03it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 25.29it/s]Capturing num tokens (num_tokens=144 avail_mem=72.50 GB):  74%|███████▍  | 43/58 [00:01<00:00, 25.29it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:02<00:00, 25.29it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:02<00:00, 25.29it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:02<00:00, 25.40it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:02<00:00, 25.40it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  79%|███████▉  | 46/58 [00:02<00:00, 25.40it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  79%|███████▉  | 46/58 [00:02<00:00, 25.40it/s]

    Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.18it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.18it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.18it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.18it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  90%|████████▉ | 52/58 [00:02<00:00, 26.14it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  90%|████████▉ | 52/58 [00:02<00:00, 26.14it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:02<00:00, 26.14it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:02<00:00, 26.14it/s]

    Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  90%|████████▉ | 52/58 [00:02<00:00, 26.14it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  97%|█████████▋| 56/58 [00:02<00:00, 27.23it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  97%|█████████▋| 56/58 [00:02<00:00, 27.23it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  97%|█████████▋| 56/58 [00:02<00:00, 27.23it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:02<00:00, 23.01it/s]


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
    Generated text:  Nathan Dede and I'm a passionate non-profit, digital artist, and creative programmer, working with people, technology, and creative spirit to bring hope, connection, and connection to others. I believe in the power of creativity and technology to bring people together. I am pursuing my MFA at the University of Southern California and have been an artist for over 15 years. My work often reflects the human experience and is focused on the intersection of technology, art, and community. I believe that technology can be used to help make the world a better place by bringing people together, promoting creativity, and promoting education. I strive to be
    ===============================
    Prompt: The president of the United States is
    Generated text:  represented by a 42-member committee, which is the largest possible committee in the United States. The vice president is represented by a committee of 21 members. What is the smallest possible number of members the second largest committee of the United States might have? The problem involves finding the smallest possible number of members in the second largest committee of the United States, given that the largest possible committee has a committee of 42 members and the vice president is represented by a committee of 21 members.
    
    To solve this, we need to consider the constraints and find a way to form the second largest committee, starting from the largest committee
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. For many years, the climate in Paris was dry and cold. But now it's changing. Scientists say the climate in Paris is changing due to the planet warming. That means the air gets warmer and the temperature rises. Scientists also say the changes will get worse. Paris is facing a serious problem now. The city needs to change its energy use. The change in energy use will change the weather. We can expect to see more rain. Our city can expect to be wetter. We can expect to see longer and wetter days. We can expect to have a cooler spring. We can expect to have a warmer summer.
    ===============================
    Prompt: The future of AI is
    Generated text:  going to be based on the robot's ability to understand language. This is a fundamental part of the development of AI.
    We have been using computers for thousands of years to communicate. In the 19th century, many early inventions relied on the ability of a user to control a computer. In the early 20th century, computers were used as tools for measuring, photographing, and analyzing matter and images. In the 1960s, computers were used for the research of biology, chemistry, and physics. In the 1980s, computers were used for image processing and scientific research. In


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short, neutral self-introduction sentence]. I'm a [insert a short, neutral self-introduction sentence]. I'm a [insert a short, neutral self-introduction sentence]. I'm a [insert a short, neutral self-introduction sentence]. I'm a [insert a short, neutral self-introduction sentence]. I'm a [insert a short, neutral self-introduction sentence]. I'm a [insert a short,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also the birthplace of French literature, cinema, and music, and is a major center for art, culture, and politics. Paris is a vibrant and diverse city with a rich history and a strong sense of French identity. It is a popular tourist destination and a major economic and cultural hub in Europe. The city is also home to many international organizations and institutions, including the European Parliament and the United Nations. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This could lead to the creation of more efficient and cost-effective solutions, but it could also lead to job displacement for humans.
    
    2. AI ethics and privacy: As AI technology becomes more advanced, there will be a need to address the ethical implications of AI, such as privacy concerns and the potential for AI to be used for malicious purposes
    


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
    Generated text:  [Name] and I am a professional freelance writer. I specialize in creating engaging and visually stunning content for various industries including [Industry]. My writing style is approachable, and I enjoy crafting content that is not only informative but also relatable. I am always eager to learn new writing techniques and share my expertise with others who need it. How can I best connect with you in the future? You can reach me through [email address] or [phone number]. Your feedback is greatly appreciated. [Name] [Contact Information] I look forward to hearing from you! Hello, my name is [Name] and I am a professional freelance
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and is located in the south of the country. It is the capital of France, the largest metropolitan area in Europe and the third largest city in the world. Paris is renowned for its museums, landmarks, and historic architecture, such as Notre-Dame Cathedral and the Eiffel Tower. It is also known for its vibrant culture, including the annual Eiffel Tower Festival, which is celebrated every July. The city is also home to several important landmarks such as Montmartre, the Latin Quarter, and the Champs-Élysées. Paris is known for its love of food and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  fascinating, and there are many possible paths it could take. Here are some potential trends that could shape the AI landscape in the coming years:
    
    1. Increased autonomy: As AI becomes more capable, it could become more self-aware and able to make decisions based on multiple factors. This could lead to a range of new applications, such as driverless cars, virtual assistants, and autonomous weapons.
    
    2. Enhanced privacy: AI is becoming increasingly powerful, but it also raises concerns about privacy and data security. There could be new tools and technologies that could help address these issues, such as blockchain-based privacy-preserving AI.
    
    3. Improved language understanding


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

    'm

     a

     [

    Your

     Profession

    ]

     with

     [

    Your

     Years

     in

     Industry

    ]

     years

     of

     experience

     in

     [

    Your

     Industry

    ].

     I

    'm

     a

     self

    -st

    arter

    ,

     always

     eager

     to

     learn

    ,

     and

     I

    'm

     passionate

     about

     [

    Your

     Passion

    ].


    As

     a

     [

    Your

     Profession

    ]

     with

     [

    Your

     Years

     in

     Industry

    ]

     years

     of

     experience

     in

     [

    Your

     Industry

    ],

     I

    'm

     always

     striving

     to

     be

     the

     best

    .

     I

    'm

     a

     strong

     communicator

    ,

     able

     to

     take

     on

     both

     a

     leader

     and

     team

     player

     role

     and

     I

    'm

     always

     willing

     to

     learn

     new

     skills

    .

     I

     believe

     that

     true

     success

     is

     not

     just

     about

     reaching

     goals

    ,

     but

     about

     creating

     them

    .

     I

    'm

     dedicated

     to

     finding

     innovative

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

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

     Lou

    vre

     Museum

    ,

     and

     Notre

    -D

    ame

     Cathedral

    ,

     as

     well

     as

     its

     renowned

     restaurants

    ,

     cafes

    ,

     and

     art

     scene

    .

     France

    's

     capital

     city

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Light

    "

     for

     its

     op

    ulent

     architecture

     and

     bustling

     streets

    .

     It

     has

     a

     rich

     history

     dating

     back

     thousands

     of

     years

     and

     was

     the

     capital

     of

     France

     for

     almost

     

    1

    5

    0

    0

     years

    .

     Paris

     is

     a

     culturally

     diverse

     and

     cosm

    opolitan

     city

     with

     a

     strong

     sense

     of

     French

     identity

     and

     influence

    .

     Its

     cuisine

    ,

     fashion

    ,

     and

     music

     also

     make

     it

     a

     popular

     tourist

     destination

    .

     Paris

     is

     a

     gateway

     to

     France

    's

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     continue

     to

     evolve

     and

     develop

     rapidly

    ,

     driven

     by

     advances

     in

     computing

     power

    ,

     data

    ,

     and

     machine

     learning

     algorithms

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     automation

     and

     robotics

    :

     More

     and

     more

     AI

    -powered

     systems

     will

     become

     autonomous

    ,

     performing

     tasks

     such

     as

     manufacturing

    ,

     maintenance

    ,

     and

     transportation

    .
    


    2

    .

     Integration

     of

     AI

     into

     everyday

     life

    :

     AI

     will

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     from

     voice

     assistants

     to

     self

    -driving

     cars

    .
    


    3

    .

     Development

     of

     ethical

     AI

    :

     As

     AI

     technology

     becomes

     more

     advanced

    ,

     ethical

     concerns

     will

     become

     more

     prominent

    ,

     such

     as

     issues

     of

     bias

    ,

     privacy

    ,

     and

     responsibility

    .
    


    4

    .

     Development

     of

     AI

     for

     human

     knowledge

     and

     creativity

    



```python
llm.shutdown()
```
