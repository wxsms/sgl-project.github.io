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
    [2026-04-16 06:02:08] `torch_dtype` is deprecated! Use `dtype` instead!
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.13it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.12it/s]


    2026-04-16 06:02:11,994 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-16 06:02:11] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:30,  2.63s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:09,  5.17it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:09,  5.17it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:09,  5.17it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:02<00:09,  5.17it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:02<00:09,  5.17it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:02<00:09,  5.17it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:02<00:09,  5.17it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:02<00:09,  5.17it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:02<00:09,  5.17it/s]Compiling num tokens (num_tokens=1792):  16%|█▌        | 9/58 [00:02<00:09,  5.17it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:02<00:03, 12.53it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:02<00:03, 12.53it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:02<00:03, 12.53it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:02<00:03, 12.53it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.53it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.53it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.53it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.53it/s]

    Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:03<00:03, 12.53it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 19.80it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 19.80it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 19.80it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 19.80it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 19.80it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 19.80it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 19.80it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 25.02it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 25.02it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 25.02it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 25.02it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 25.02it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 25.02it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 25.02it/s]

    Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:03<00:01, 25.02it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 32.04it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:03<00:00, 38.91it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:03<00:00, 38.91it/s]

    Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:03<00:00, 38.91it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:03<00:00, 48.36it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:03<00:00, 48.36it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:03<00:00, 48.36it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:03<00:00, 48.36it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:02, 19.02it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.51it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.51it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.68it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.68it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.68it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.68it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.68it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.71it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.69 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.71it/s] Capturing num tokens (num_tokens=896 avail_mem=76.69 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=832 avail_mem=76.69 GB):  34%|███▍      | 20/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=832 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=768 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=704 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=640 avail_mem=76.66 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=576 avail_mem=76.18 GB):  41%|████▏     | 24/58 [00:00<00:01, 33.96it/s]Capturing num tokens (num_tokens=576 avail_mem=76.18 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.27it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.27it/s]

    Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.27it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.27it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.27it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  48%|████▊     | 28/58 [00:00<00:00, 35.27it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:00<00:00, 39.00it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.00it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.00it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.00it/s]

    Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  57%|█████▋    | 33/58 [00:01<00:00, 39.00it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.75it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.75it/s]Capturing num tokens (num_tokens=208 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.75it/s]Capturing num tokens (num_tokens=192 avail_mem=76.00 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.75it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.75it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.40it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.40it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.40it/s]

    Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.40it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.40it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 34.40it/s] Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  81%|████████  | 47/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=48 avail_mem=75.97 GB):  81%|████████  | 47/58 [00:01<00:00, 34.72it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  81%|████████  | 47/58 [00:01<00:00, 34.72it/s]

    Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.75it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.75it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.75it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.75it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.75it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 34.75it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.92it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.92it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 37.92it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 35.02it/s]


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
    Generated text:  Zhehao Ma, and I am a graduate student in the Information Science and Engineering Department at the University of Waterloo. I am currently a Research Fellow at the University of Waterloo. Before joining Waterloo, I was a Postdoctoral Researcher at the University of Toronto. My PhD research investigates the application of artificial neural networks for time series analysis.
    I am also a member of the Computational Science and Engineering (CSE) working group at Waterloo. I am a member of the Data Science and Machine Learning working group at the University of Toronto. I am a member of the AI and Data Science 2018 Summer Program at the
    ===============================
    Prompt: The president of the United States is
    Generated text:  in Washington, D. C., where he is in a wheelchair. He is a man and 42 years old.
    The president of the United States, who is currently in Washington D. C., where he is a wheelchair-dependent individual, is a man and 42 years old. President Donald Trump, also known as the "Trumpist," has been in this position since 2011. He served two terms in the White House, including as the 45th President from 2017 to 2021. Trump has faced numerous controversies and challenges, including his controversial statements about the 
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is situated at the mouth of the Seine river, and the Seine runs through the city. The city of Paris is famous for its museums, its art, and its castles. It is also well known for its love of the classical period, particularly of the French Renaissance and the French Baroque. Other important institutions include the Musée Rodin, the Musée d'Orsay, the Musée d'Orsay, and the Musée de l'Orangerie. The National Library of France also houses the National Library of France, the National Museum of Natural History, and the Musée de l
    ===============================
    Prompt: The future of AI is
    Generated text:  exciting, with endless possibilities. However, it is essential to understand that AI is still a relatively new field and its applications have only just begun. As such, it is important to have a basic understanding of the different types of AI and their applications. One of the most popular types of AI is machine learning, which involves training a computer algorithm to learn from data and improve its performance over time. Other popular types of AI include deep learning, which involves using neural networks to solve complex problems, and natural language processing, which involves enabling computers to understand and interpret human language.
    Machine learning is a key component of many AI applications, such as image


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Skill/Ability] who has been [Number of Years] years in the field of [Field of Interest]. I'm a [Favorite Activity] enthusiast and I love [Favorite Food/Drink/Activity]. I'm a [Favorite Book/TV Show/Video Game] lover and I enjoy [Favorite Music/Artist]. I'm a [Favorite Movie/Book/TV Show/Video Game] buff and I love [Favorite Sport/Activity/Travel/Other]. I'm a [Favorite Person/Company/Place] enthusiast and I love
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament and the French National Museum. Paris is a bustling city with a rich cultural heritage and is a popular tourist destination. The city is known for its cuisine, fashion, and art scene. It is also home to the French Parliament and the French National Museum. Paris is a vibrant and dynamic city that is known for its unique blend of history, culture, and modernity. The city is also known for its iconic landmarks and is a popular tourist destination. The French
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems.
    
    2. Greater emphasis on ethical and social implications: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical and social implications. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Increased focus on AI ethics and safety: As AI becomes more integrated with human intelligence, there will be increased focus on ethical and safety concerns. This
    


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
    Generated text:  [Name]. I am an AI language model programmed to assist people in generating text with natural language. I am here to help with a wide range of questions and problems. Whether you need help with a question or a solution, I am here to assist you. So, what's your question or problem? Let's get started! 
    
    Please fill out the form below. I will get back to you as soon as possible. 
    
    Name: [Name] (or whatever you want)
    Email: [Your email address]
    Phone: [Your phone number]
    About you: [Describe your personality, skills, or interests]
    [Optional] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower and Notre Dame Cathedral. It is also renowned for its rich cultural heritage, vibrant nightlife, and role in the French Revolution and the French Revolution. Paris is a major hub for arts, music, and fashion, and is home to the headquarters of many major French companies. With its beautiful French architecture and iconic monuments, Paris is a cultural and historical gem of the world. Can you provide a French sentence that incorporates the concept of "eiffel tower" in the context of the capital of France?
    
    Sure! Here's a French sentence incorporating the concept of "eiff
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and exciting, with many potential areas of development and innovation. Here are some potential future trends in AI:
    
    1. Cognitive Augmentation: As AI continues to learn and improve, it may be possible to enhance its capabilities by incorporating more human-like behaviors and emotions into its programming.
    
    2. Personalized AI: As AI is used in various industries, it may become increasingly personalized to meet the specific needs and preferences of individual users.
    
    3. Ethical AI: As more and more AI systems are deployed in various settings, it may become necessary to address ethical issues such as bias, transparency, and accountability.
    
    4. Autonomous Systems: As AI


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

    Type

     of

     Person

    ].

     I

     enjoy

     [

    What

     You

     Do

    /

    What

     You

     Do

     Well

    ].

     I

     love

     [

    Something

     That

     Takes

     the

     Moment

    ].

     I

    'm

     [

    What

     You

     Like

     About

     Yourself

    ].

     I

     hope

     to

     be

     a

     [

    What

     You

     Want

     to

     be

    ].

     How

     do

     you

     feel

     about

     [

    Career

    /

    Interest

    /

    Personal

     Goal

    ]?

     (

    e

    .g

    .

     [

    What

     You

     Do

     Well

    ,

     What

     You

     Can

     Do

     Well

    ,

     etc

    .

    ])

     I

    'm

     excited

     to

     meet

     you

     and

     ask

     you

     a

     few

     questions

     about

     yourself

    .

     Is

     there

     a

     particular

     aspect

     of

     your

     life

     that

     you

    're

     particularly

     interested

     in

    ?

     (

    e

    .g

    .

     [

    What

     You

     Want

     to

     be

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Mediterranean

     coast

     of

     the

     country

     and

     one

     of

     the

     most

     important

     cities

     in

     Europe

    .

     It

     has

     a

     rich

     cultural

     history

    ,

     with

     influences

     from

     both

     French

     and

     Mediterranean

     cultures

    .

     The

     city

     is

     known

     for

     its

     lively

     city

     life

    ,

     its

     famous

     E

    iff

    el

     Tower

    ,

     and

     its

     delicious

     cuisine

    .

     Paris

     is

     also

     home

     to

     a

     number

     of

     UNESCO

     world

     heritage

     sites

     and

     is

     an

     important

     economic

     and

     political

     center

    .

     It

     is

     a

     city

     that

     is

     constantly

     evolving

    ,

     with

     ongoing

     efforts

     to

     preserve

     and

     promote

     its

     unique

     identity

    .

     It

     is

     widely

     considered

     one

     of

     the

     most

     fascinating

     and

     vibrant

     cities

     in

     the

     world

    .

     Paris

    ,

     officially

     known

     as

     the

     "

    City

     of

     Light

    ,"

     is

     a

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bound

     to

     be

     unpredictable

    ,

     but

     here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     industry

    :
    


    1

    .

     Autonomous

     robots

    :

     With

     advancements

     in

     robotics

     and

     artificial

     intelligence

    ,

     we

     can

     expect

     to

     see

     more

     autonomous

     robots

     that

     can

     perform

     tasks

     in

     a

     wide

     range

     of

     fields

    ,

     such

     as

     manufacturing

    ,

     delivery

    ,

     and

     transportation

    .
    


    2

    .

     Improved

     machine

     learning

     algorithms

    :

     As

     machine

     learning

     algorithms

     get

     more

     sophisticated

    ,

     we

     can

     expect

     to

     see

     more

     advanced

     models

     that

     can

     learn

     from

     large

     amounts

     of

     data

     and

     make

     more

     accurate

     predictions

     and

     decisions

    .
    


    3

    .

     Enhanced

     security

    :

     As

     we

     become

     more

     reliant

     on

     AI

     in

     our

     daily

     lives

    ,

     we

     may

     see

     an

     increase

     in

     the

     use

     of

     AI

    -based

     security

     systems

     that

    



```python
llm.shutdown()
```
