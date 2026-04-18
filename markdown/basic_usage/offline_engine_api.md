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


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-18 09:13:06] `torch_dtype` is deprecated! Use `dtype` instead!
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.48it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.48it/s]


    2026-04-18 09:13:10,870 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-18 09:13:10] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:31,  2.66s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.88it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  5.14it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  5.14it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  5.14it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:09,  5.14it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:09,  5.14it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:09,  5.14it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:09,  5.14it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:02<00:09,  5.14it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:02<00:09,  5.14it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:02<00:03, 11.68it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:02<00:03, 11.68it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:02<00:03, 11.68it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:03<00:03, 11.68it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:03<00:03, 11.68it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:03<00:03, 11.68it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:03<00:03, 11.68it/s]

    Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:02, 17.01it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:02, 17.01it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:02, 17.01it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:02, 17.01it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:02, 17.01it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:03<00:02, 17.01it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:03<00:02, 17.01it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:03<00:02, 17.01it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 23.73it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 23.73it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 23.73it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 23.73it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 23.73it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 23.73it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 23.73it/s]

    Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 23.73it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:03<00:00, 30.24it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 34.43it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 34.43it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 34.43it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 34.43it/s]

    Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 34.43it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 34.43it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 34.43it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 36.19it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 36.19it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 36.19it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 36.19it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 36.19it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 36.19it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 36.19it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 40.96it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 40.96it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 40.96it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 40.96it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.64it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=49.84 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=49.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=49.81 GB):   3%|▎         | 2/58 [00:00<00:03, 16.83it/s]Capturing num tokens (num_tokens=7168 avail_mem=49.81 GB):   3%|▎         | 2/58 [00:00<00:03, 16.83it/s]Capturing num tokens (num_tokens=6656 avail_mem=49.81 GB):   3%|▎         | 2/58 [00:00<00:03, 16.83it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=49.81 GB):   3%|▎         | 2/58 [00:00<00:03, 16.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=49.81 GB):   9%|▊         | 5/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=49.80 GB):   9%|▊         | 5/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=49.80 GB):   9%|▊         | 5/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=49.80 GB):   9%|▊         | 5/58 [00:00<00:02, 19.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=49.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=4096 avail_mem=49.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=49.80 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.15it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=49.79 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=49.79 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.15it/s]Capturing num tokens (num_tokens=3328 avail_mem=49.79 GB):  21%|██        | 12/58 [00:00<00:01, 25.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=49.78 GB):  21%|██        | 12/58 [00:00<00:01, 25.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=49.78 GB):  21%|██        | 12/58 [00:00<00:01, 25.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=49.78 GB):  21%|██        | 12/58 [00:00<00:01, 25.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=49.78 GB):  21%|██        | 12/58 [00:00<00:01, 25.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=49.78 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=2048 avail_mem=49.77 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.94it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=49.77 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=49.76 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=1280 avail_mem=49.76 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.94it/s]Capturing num tokens (num_tokens=1280 avail_mem=49.76 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=49.74 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.26it/s]Capturing num tokens (num_tokens=960 avail_mem=49.72 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.26it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=49.72 GB):  34%|███▍      | 20/58 [00:00<00:01, 27.26it/s]Capturing num tokens (num_tokens=896 avail_mem=49.72 GB):  40%|███▉      | 23/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=832 avail_mem=49.72 GB):  40%|███▉      | 23/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=768 avail_mem=49.71 GB):  40%|███▉      | 23/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=704 avail_mem=49.71 GB):  40%|███▉      | 23/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=640 avail_mem=49.71 GB):  40%|███▉      | 23/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=576 avail_mem=49.71 GB):  40%|███▉      | 23/58 [00:01<00:01, 26.24it/s]Capturing num tokens (num_tokens=576 avail_mem=49.71 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.70it/s]Capturing num tokens (num_tokens=512 avail_mem=49.69 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.70it/s]Capturing num tokens (num_tokens=480 avail_mem=49.71 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.70it/s]

    Capturing num tokens (num_tokens=448 avail_mem=49.71 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.70it/s]Capturing num tokens (num_tokens=416 avail_mem=49.71 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.70it/s]Capturing num tokens (num_tokens=416 avail_mem=49.71 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=384 avail_mem=49.70 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=352 avail_mem=49.70 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=320 avail_mem=49.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=288 avail_mem=49.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=256 avail_mem=49.69 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.44it/s]Capturing num tokens (num_tokens=256 avail_mem=49.69 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=240 avail_mem=49.69 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=224 avail_mem=49.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=208 avail_mem=49.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.50it/s]

    Capturing num tokens (num_tokens=192 avail_mem=49.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=176 avail_mem=49.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=160 avail_mem=49.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 36.50it/s]Capturing num tokens (num_tokens=160 avail_mem=49.67 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=144 avail_mem=49.67 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=128 avail_mem=49.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=112 avail_mem=49.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=96 avail_mem=49.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.86it/s] Capturing num tokens (num_tokens=80 avail_mem=49.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.86it/s]Capturing num tokens (num_tokens=80 avail_mem=49.65 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=64 avail_mem=49.65 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=48 avail_mem=49.65 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.18it/s]

    Capturing num tokens (num_tokens=32 avail_mem=49.65 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=28 avail_mem=49.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=24 avail_mem=49.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 43.18it/s]Capturing num tokens (num_tokens=24 avail_mem=49.64 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.99it/s]Capturing num tokens (num_tokens=20 avail_mem=49.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.99it/s]Capturing num tokens (num_tokens=16 avail_mem=49.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.99it/s]Capturing num tokens (num_tokens=12 avail_mem=49.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.99it/s]Capturing num tokens (num_tokens=8 avail_mem=49.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.99it/s] Capturing num tokens (num_tokens=4 avail_mem=49.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.99it/s]Capturing num tokens (num_tokens=4 avail_mem=49.62 GB): 100%|██████████| 58/58 [00:01<00:00, 34.93it/s]


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
    Generated text:  Jonathan. I’m a 30 year old female and I have been diagnosed with bipolar disorder. What does it feel like to have it? I don’t want to live life like I should be living, but I’m afraid to tell anyone. I am very nervous and don’t have a clue how to deal with it. My family really supports me and treats me the best way they can. I have been given medication to help me manage my disorder. I’ve even learned a few things in my life from those treatments. I’m not sure how to tell anyone, but I really want to do it. Please, I need help
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office. There have been many presidents throughout US history. The first president was George Washington. He served from 1789 to 1797. How many terms would he serve? To determine how many terms George Washington would serve as the President of the United States, we need to understand the presidential term length and how it applies to George Washington's time in office.
    
    1. **Identify the Presidential Term Length:**
       - The Presidential term is 4 years.
    
    2. **Determine the Number of Terms:**
       - Washington served from 1789 to 1797.
    
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Nice
    C. Bruges
    D. Bordeaux
    
    The correct answer is A. Paris. France's capital is Paris. The other cities listed are not in France:
    
    B. Nice: Located in France
    C. Bruges: Located in Belgium
    D. Bordeaux: Located in France
    
    None of the other cities listed are in France. Paris is the capital of France. The other options refer to other countries, not the capital of France. Therefore, the correct answer is Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  in process, and the people behind it are taking the lead. The next decade will witness a revolution in how we use AI, from social media to autonomous vehicles. Some are pushing back against the growth of AI, while others are embracing it. The following are some of the most interesting trends currently shaping the future of AI.
    The role of AI in the medical field is growing steadily, as more and more patients are sharing their medical information online. The data collected can be used to create personalized treatment plans for individuals, as well as to improve the diagnosis and treatment of diseases.
    In addition, AI is being used to predict disease outbreaks and prevent


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


    Generated text:  [Name] and I am a [occupation] who has been [number of years] in the industry. I am passionate about [reason for passion], and I am always looking for ways to [action or goal]. I am a [type of person] who is [positive or negative] about life, and I am always looking for ways to [action or goal]. I am a [type of person] who is [positive or negative] about life, and I am always looking for ways to [action or goal]. I am a [type of person] who is [positive or negative] about life, and I am always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major economic and political center in Europe. It is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The city is home to many international institutions and organizations, including the European Parliament, the United Nations, and the International Olympic Committee. Paris is also known for its cuisine, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more sophisticated and adaptive AI systems that can learn from their environment and improve their performance over time.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased scrutiny of AI systems that are designed to harm or mislead humans.
    
    3.
    


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
    Generated text:  [name], and I'm a [occupation] at [company]. I'm currently working on [current project]. What can you tell me about yourself?
    
    [Name] is a [occupation] at [company]. I'm currently working on [current project]. I'm a [background] personality who is known for my [adjective], [noun] and [verb]. I believe in [value] principles and strive to be a [mission] person. I'm passionate about [objective], and I'm always looking for [future goal]. [Name] is a [occupation] at [company]. I'm currently working on [current
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement encapsulates the central role of Paris in French politics, culture, and history, as it is the most populous city in France and has been the seat of the French government since 1871. The city is renowned for its architecture, art, and cuisine, and is a major center of international trade and diplomacy. Paris has been described as "the City of Light" and "the Eternal City," reflecting its status as the "most beautiful city in the world" according to the Guinness World Records. It is also home to the Louvre Museum and the Eiffel Tower, both iconic landmarks. Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it will become more integrated with human intelligence and cognitive abilities, allowing it to perform tasks more accurately and efficiently. This will be particularly useful in areas like healthcare, where AI can help doctors make more accurate diagnoses and treatment plans.
    
    2. Development of more ethical and accountable AI: As AI becomes more advanced, there will be increased scrutiny of its development and deployment. Developers will need to ensure that AI is used ethically and responsibly, taking into account the potential impact on individuals and society as a whole.
    
    3. Greater emphasis on


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

    'm

     a

     [

    Occup

    ation

    ]

     who

     was

     introduced

     to

     you

     in

     [

    Your

     Introduction

    ].

     I

    'm

     a

     dedicated

     [

    Job

     Title

    ]

     who

     has

     always

     loved

     [

    Why

     You

     Became

     a

     [

    Job

     Title

    ]],

     and

     I

    'm

     always

     up

     for

     [

    What

     You

     Can

     Do

     Next

    ],

     no

     matter

     the

     situation

    .

     How

     can

     I

     assist

     you

     today

    ?

     Remember

    ,

     everyone

     is

     different

    ,

     so

     tailor

     your

     response

     to

     reflect

     your

     unique

     character

    .

     Good

     luck

    !

     [

    Start

     your

     self

    -int

    roduction

    ]

     Alright

    ,

     let

    's

     get

     started

    !

     Let

    's

     see

     how

     much

     I

     can

     do

     for

     you

     today

    .

     [

    End

     your

     self

    -int

    roduction

    ]

     Keep

     on

     the

     roll

    !

     [

    Start

     your

     self

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

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

    ,

     Lou

    vre

     Museum

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Le

     Mou

    le

    .

     The

     city

     is

     also

     famous

     for

     its

     rich

     history

     and

     culture

    ,

     with

     several

     important

     museums

    ,

     theaters

    ,

     and

     festivals

    .

     Paris

     is

     a

     major

     cultural

     and

     economic

     center

     and

     is

     often

     referred

     to

     as

     the

     "Paris

     of

     the

     world

    ".

     
    


    French

     is

     the

     official

     language

     of

     France

    ,

     and

     it

     is

     the

     language

     of

     education

     and

     government

     in

     the

     country

    .

     Paris

     has

     a

     diverse

     population

    ,

     including

     French

    ,

     African

    ,

     and

     other

     international

     communities

    .

     The

     city

     is

     a

     popular

     tourist

     destination

    ,

     with

     millions

     of

     tourists

     visiting

     each

     year

     for

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     rapid

     and

     un

    quant

    ifiable

     transformation

    .

     AI

     will

     likely

     continue

     to

     grow

     and

     evolve

     at

     an

     unprecedented

     pace

    ,

     bringing

     about

     a

     wide

     range

     of

     new

     developments

     and

     applications

    .

     Some

     possible

     trends

     include

    :
    


    1

    .

     Increased

     depth

     and

     complexity

    :

     AI

     is

     already

     at

     the

     forefront

     of

     many

     domains

    ,

     from

     healthcare

     and

     finance

     to

     manufacturing

     and

     transportation

    .

     As

     the

     technology

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

     deeper

     integration

     of

     AI

     with

     other

     industries

    ,

     leading

     to

     even

     more

     complex

     and

     advanced

     systems

    .
    


    2

    .

     Personal

    ization

    :

     AI

     is

     already

     used

     to

     personalize

     the

     way

     we

     interact

     with

     the

     world

     around

     us

    .

     As

     AI

     becomes

     more

     sophisticated

    ,

     we

     can

     expect

    



```python
llm.shutdown()
```
