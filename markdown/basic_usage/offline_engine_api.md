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
    [2026-04-22 09:58:32] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.70it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.68it/s]


    2026-04-22 09:58:37,431 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-22 09:58:37] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:39,  2.80s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.26it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:03<00:23,  2.26it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.60it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.51it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.51it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.51it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.51it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.51it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.51it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.51it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.51it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.51it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.51it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.42it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.10it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.10it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.10it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 38.72it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 49.08it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=121.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=121.63 GB):   3%|▎         | 2/58 [00:00<00:03, 17.95it/s]Capturing num tokens (num_tokens=7168 avail_mem=121.62 GB):   3%|▎         | 2/58 [00:00<00:03, 17.95it/s]Capturing num tokens (num_tokens=6656 avail_mem=119.15 GB):   3%|▎         | 2/58 [00:00<00:03, 17.95it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=119.15 GB):   7%|▋         | 4/58 [00:00<00:02, 18.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=116.36 GB):   7%|▋         | 4/58 [00:00<00:02, 18.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=116.36 GB):   7%|▋         | 4/58 [00:00<00:02, 18.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.36 GB):   7%|▋         | 4/58 [00:00<00:02, 18.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=116.36 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=4608 avail_mem=116.36 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=4096 avail_mem=116.36 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=116.35 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=116.35 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.59it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=116.35 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=3328 avail_mem=116.34 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=3072 avail_mem=116.34 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=116.34 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=2560 avail_mem=116.33 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.33 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.97it/s]Capturing num tokens (num_tokens=2304 avail_mem=116.33 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.06it/s]Capturing num tokens (num_tokens=2048 avail_mem=116.33 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.06it/s]Capturing num tokens (num_tokens=1792 avail_mem=116.32 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.06it/s]Capturing num tokens (num_tokens=1536 avail_mem=116.32 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.06it/s]Capturing num tokens (num_tokens=1280 avail_mem=116.32 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.06it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=116.30 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.06it/s]Capturing num tokens (num_tokens=1024 avail_mem=116.30 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.97it/s]Capturing num tokens (num_tokens=960 avail_mem=116.31 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.97it/s] Capturing num tokens (num_tokens=896 avail_mem=116.31 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.97it/s]Capturing num tokens (num_tokens=832 avail_mem=116.30 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.97it/s]Capturing num tokens (num_tokens=768 avail_mem=116.30 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.97it/s]Capturing num tokens (num_tokens=704 avail_mem=116.30 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.97it/s]Capturing num tokens (num_tokens=704 avail_mem=116.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.15it/s]Capturing num tokens (num_tokens=640 avail_mem=116.29 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.15it/s]Capturing num tokens (num_tokens=576 avail_mem=116.29 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.15it/s]Capturing num tokens (num_tokens=512 avail_mem=116.28 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.15it/s]

    Capturing num tokens (num_tokens=480 avail_mem=116.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.15it/s]Capturing num tokens (num_tokens=448 avail_mem=116.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.15it/s]Capturing num tokens (num_tokens=448 avail_mem=116.30 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=416 avail_mem=116.29 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=384 avail_mem=116.29 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=352 avail_mem=116.29 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=320 avail_mem=116.28 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.39it/s]Capturing num tokens (num_tokens=288 avail_mem=116.28 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.39it/s]Capturing num tokens (num_tokens=288 avail_mem=116.28 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=256 avail_mem=116.28 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=240 avail_mem=116.28 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.92it/s]

    Capturing num tokens (num_tokens=224 avail_mem=116.27 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=208 avail_mem=116.27 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=192 avail_mem=116.27 GB):  62%|██████▏   | 36/58 [00:01<00:00, 39.92it/s]Capturing num tokens (num_tokens=192 avail_mem=116.27 GB):  71%|███████   | 41/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=176 avail_mem=116.26 GB):  71%|███████   | 41/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=160 avail_mem=116.26 GB):  71%|███████   | 41/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=144 avail_mem=116.26 GB):  71%|███████   | 41/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=128 avail_mem=116.25 GB):  71%|███████   | 41/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=112 avail_mem=116.25 GB):  71%|███████   | 41/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=112 avail_mem=116.25 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=96 avail_mem=116.25 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.45it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=116.24 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=64 avail_mem=116.24 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=48 avail_mem=116.24 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=32 avail_mem=116.23 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.45it/s]Capturing num tokens (num_tokens=32 avail_mem=116.23 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=28 avail_mem=116.23 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=24 avail_mem=116.23 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=20 avail_mem=116.22 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=16 avail_mem=116.22 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.20it/s]Capturing num tokens (num_tokens=12 avail_mem=116.22 GB):  88%|████████▊ | 51/58 [00:01<00:00, 41.20it/s]

    Capturing num tokens (num_tokens=12 avail_mem=116.22 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=8 avail_mem=116.21 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.11it/s] Capturing num tokens (num_tokens=4 avail_mem=116.21 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.11it/s]Capturing num tokens (num_tokens=4 avail_mem=116.21 GB): 100%|██████████| 58/58 [00:01<00:00, 37.31it/s]


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
    Generated text:  Igor and I'm the owner of the website. I specialize in the assignment of research questions to students and I have helped many students achieve their academic goals. I am a retired professor from the University of Toronto and a former professor at the University of Ottawa. My area of research interests is general philosophy of language. I also teach philosophy of language at the University of Ottawa and I'm the editor of the journal "Philosophy of Language" and a member of the editorial board of "Philosophy of the Sciences" and "Philosophy and Phenomenological Research".
    My long-time student has been one of the top students in my
    ===============================
    Prompt: The president of the United States is
    Generated text:  very busy preparing for the upcoming presidency. So, he decided to travel to Europe for a week and back to the United States to have a vacation. The cost of a one-way trip is $4000. The president will travel to Europe 4 times and will stay in Europe for 2 weeks. The president wants to take advantage of the president’s vacation and go on a short trip to Europe to visit his mother, which costs $500. How much will the president spend in total for his presidency, including the one-way trip to Europe, the one-way trip back to the United States, the two weeks he
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris
    
    B. Berlin
    
    C. London
    
    D. Moscow
    
    1. **Understanding the Question**: The question asks us to identify the capital of France and provide the correct answer.
    
    2. **Identifying the Capital of France**:
       - France is the largest country in Europe by area.
       - It borders three oceans: the Atlantic, the Mediterranean, and the English Channel.
       - It is located on the continent of Europe.
       - It is known for its cultural, historical, and artistic heritage, as well as for its rich cuisine, language, and architecture.
    
    3. **Analyzing the Options**
    ===============================
    Prompt: The future of AI is
    Generated text:  now, and it is even more exciting for the industry. The number of new AI technologies and innovations is growing at a massive pace. What are the most promising areas for AI, and how can the industry adapt to this new era of technology? For instance, how can we harness the power of AI to create a more sustainable future? Can we leverage AI to improve healthcare outcomes, reduce healthcare costs, and enhance the quality of life for people around the world? Ultimately, what challenges do we face in developing AI that is ethical, equitable, and sustainable? The future of AI is exciting, and the industry can take advantage of it. By


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill/Ability] who has always been [Positive Qualities] and [Negative Qualities]. I'm passionate about [What I Love to Do]. I'm always looking for [What I Want to Learn/Discover]. I'm [What I'm Known for]. I'm [What I'm Looking for in a Partner]. I'm [What I'm Looking for in a Job]. I'm [What I'm Looking for in a Friend]. I'm [What I'm Looking for in a Family]. I'm [What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the world by population. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also famous for its rich history, art, and cuisine. Paris is a major cultural and economic center, and it is home to many world-renowned museums, theaters, and restaurants. It is also a popular tourist destination, attracting millions of visitors each year. Paris is a vibrant and dynamic city that is a must-visit for anyone interested in French culture and history. 
    
    Paris is the capital of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends include:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to diagnose and treat diseases, and it has the potential to revolutionize the field. AI-powered diagnostic tools, such as AI-powered X-rays and AI-powered pathology analysis, could lead to more accurate diagnoses and faster treatment of diseases.
    
    2. Increased use of AI in transportation: AI is already being used in transportation to improve safety, efficiency, and reduce traffic congestion. AI-powered autonomous vehicles, such as those developed by Tesla
    


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
    Generated text:  [Name] and I'm a [field] specialist. I have [number] years of experience in [field], and my current position is [position]. I enjoy [main activity], and I believe I have [number] years of experience in [field]. I'm always looking for new challenges and opportunities to expand my skills and knowledge. How can I be of assistance? [Name] [Experience] [Skills] [Professional Experience] [Education] [Certifications] [Professional Achievements] [Professional Background] [Experience] [Skills] [Professional Achievements] [Professional Background] [Experience] [Skills] [Professional
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is also known as the “City of Love” and is one of the most famous cities in the world. Paris is a cultural and historic center that is known for its famous landmarks, including the Eiffel Tower and the Notre Dame Cathedral. The city is also famous for its vibrant art scene and its annual couture fashion show. Paris is a world-renowned hub of finance, arts, and culture and is one of the most visited cities in the world. It is often referred to as the “City of Light” and the “City of Light” is an apt description of the city’s reputation. The city is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and rapidly evolving, with exciting possibilities and potential downsides. Some possible future trends in AI include:
    
    1. Integration of AI with other technologies: As AI becomes more integrated with other technologies, we can expect to see new applications and improvements in areas such as healthcare, transportation, and manufacturing.
    
    2. Increased focus on ethics and fairness: With the increasing amount of data being collected and analyzed by AI systems, there will be a greater emphasis on ethical considerations and fairness in the development and use of AI.
    
    3. AI development for human needs: AI is already being developed for a variety of human needs such as language processing, natural language


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

    job

     title

    ]

     at

     [

    company

    ].

     I

    've

     been

     with

     the

     company

     for

     [

    number

     of

     years

    ]

     years

    .

     I

     have

     a

     lot

     of

     experience

     working

     in

     [

    industry

    ],

     and

     I

    'm

     always

     looking

     for

     ways

     to

     [

    adv

    ocate

     or improve

    ].

     I

    'm

     a

     [

    character

     trait

    ]

     and

     I

     strive

     to

     [

    goal

     or

     achievement

    ].

     I

    'm

     known

     for

     [

    person

    ality

     trait

    ].

     I

     am

     always

     available

     to

     help

     and

     collaborate

     with

     my

     team

    .

     I

     have

     a

     good

     understanding

     of

     [

    industry

    ],

     and

     I

     thrive

     in

     [

    job

     role

    ].

     I

    'm

     a

     [

    character

     trait

    ]

     and

     I

     strive

     to

     [

    goal

     or

     achievement

    ].

     I

    'm

     known

     for

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     located

     on

     the

     Se

    ine

     River

    ,

     the

     longest

     river

     in

     the

     world

     by

     length

    ,

     and

     has

     a

     population

     of

     over

     

    2

    .

    1

     million

     people

    .
    


    Paris

     is

     an

     iconic

     city

    ,

     known

     for

     its

     stunning

     architecture

    ,

     museums

    ,

     and

     annual

     Paris

    ian

     festivals

     such

     as

     the

     E

    iff

    el

     Tower

     celebration

     and

     the

     P

    éri

    ode

     des

     L

    umi

    ères

    ,

     a

     celebration

     of

     light

    .

     It

     is

     also

     a

     center

     for

     culture

    ,

     with

     an

     active

     theater

    ,

     opera

    ,

     and

     concert

     scene

    ,

     as

     well

     as

     numerous

     museums

    ,

     art

     galleries

    ,

     and

     galleries

     of

     foreign

     artists

    .

     Paris

     has

     also

     been

     a

     home

     to

     important

     historical

     events

     and

     cultural

     movements

    ,

     including

     the

     French

     Revolution

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     rapid

     and

     significant

     changes

    ,

     both

     positive

     and

     negative

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

     Enhanced

     AI

     capabilities

    :

     With

     advancements

     in

     machine

     learning

     and

     deep

     learning

    ,

     AI

     is

     becoming

     more

     powerful

     and

     capable

     of

     performing

     tasks

     that

     were

     previously

     impossible

    .

     This

     could

     lead

     to

     new

     applications

     and

     industries

     that

     were

     previously

     unimagin

    able

    ,

     such

     as

     smart

     homes

    ,

     self

    -driving

     cars

    ,

     and

     virtual

     reality

    .
    


    2

    .

     Increased

     privacy

     concerns

    :

     As

     AI

     systems

     become

     more

     complex

     and

     sophisticated

    ,

     there

     will

     be

     increasing

     concerns

     about

     privacy

     and

     data

     security

    .

     This

     could

     lead

     to

     new

     regulations

     and

     standards

     to

     protect

     user

     data

     and

     privacy

    .
    


    3

    .

     AI

     ethics

    :

     AI

    



```python
llm.shutdown()
```
