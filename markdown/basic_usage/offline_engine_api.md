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


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-19 12:38:25] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.26it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.26it/s]


    2026-04-19 12:38:29,348 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-19 12:38:29] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:28,  2.60s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.93it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.93it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.93it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.93it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.93it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.93it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.93it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:08,  5.91it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 14.20it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 14.20it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 14.20it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 14.20it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:02<00:02, 14.20it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:02<00:02, 14.20it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:02<00:02, 14.20it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:02<00:02, 14.20it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:02<00:02, 14.20it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 14.20it/s]

    Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 22.71it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 22.71it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 22.71it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 22.71it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 22.71it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 22.71it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 22.71it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 22.71it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 22.71it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 22.71it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:03<00:01, 22.71it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 33.14it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 33.14it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:03<00:00, 33.14it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 43.77it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 43.77it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 43.77it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 43.77it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 43.77it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 43.77it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 43.77it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 43.77it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 43.77it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 43.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=49.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=49.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=49.72 GB):   3%|▎         | 2/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=7168 avail_mem=49.72 GB):   3%|▎         | 2/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=6656 avail_mem=49.71 GB):   3%|▎         | 2/58 [00:00<00:02, 18.86it/s]Capturing num tokens (num_tokens=6144 avail_mem=49.72 GB):   3%|▎         | 2/58 [00:00<00:02, 18.86it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=49.72 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=5632 avail_mem=49.71 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=5120 avail_mem=49.71 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=49.71 GB):   9%|▊         | 5/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=4608 avail_mem=49.71 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.29it/s]Capturing num tokens (num_tokens=4096 avail_mem=49.71 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.29it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=49.70 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=49.70 GB):  14%|█▍        | 8/58 [00:00<00:02, 19.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=49.70 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=3328 avail_mem=49.69 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=3072 avail_mem=49.69 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.88it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=53.57 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.88it/s]Capturing num tokens (num_tokens=2816 avail_mem=53.57 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=53.56 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=53.56 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=53.56 GB):  24%|██▍       | 14/58 [00:00<00:02, 20.09it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=53.56 GB):  29%|██▉       | 17/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=1792 avail_mem=53.55 GB):  29%|██▉       | 17/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=1536 avail_mem=53.55 GB):  29%|██▉       | 17/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=1280 avail_mem=53.54 GB):  29%|██▉       | 17/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.52 GB):  29%|██▉       | 17/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=1024 avail_mem=53.52 GB):  36%|███▌      | 21/58 [00:01<00:01, 22.28it/s]Capturing num tokens (num_tokens=960 avail_mem=53.54 GB):  36%|███▌      | 21/58 [00:01<00:01, 22.28it/s] Capturing num tokens (num_tokens=896 avail_mem=53.54 GB):  36%|███▌      | 21/58 [00:01<00:01, 22.28it/s]Capturing num tokens (num_tokens=832 avail_mem=53.53 GB):  36%|███▌      | 21/58 [00:01<00:01, 22.28it/s]Capturing num tokens (num_tokens=768 avail_mem=53.53 GB):  36%|███▌      | 21/58 [00:01<00:01, 22.28it/s]

    Capturing num tokens (num_tokens=704 avail_mem=53.53 GB):  36%|███▌      | 21/58 [00:01<00:01, 22.28it/s]Capturing num tokens (num_tokens=704 avail_mem=53.53 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.13it/s]Capturing num tokens (num_tokens=640 avail_mem=53.52 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.13it/s]Capturing num tokens (num_tokens=576 avail_mem=53.52 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.13it/s]Capturing num tokens (num_tokens=512 avail_mem=53.51 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.13it/s]Capturing num tokens (num_tokens=480 avail_mem=53.53 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.13it/s]Capturing num tokens (num_tokens=448 avail_mem=53.52 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.13it/s]Capturing num tokens (num_tokens=448 avail_mem=53.52 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=416 avail_mem=53.52 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=384 avail_mem=53.52 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=352 avail_mem=53.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=320 avail_mem=53.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.14it/s]

    Capturing num tokens (num_tokens=288 avail_mem=53.51 GB):  53%|█████▎    | 31/58 [00:01<00:00, 33.14it/s]Capturing num tokens (num_tokens=288 avail_mem=53.51 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=256 avail_mem=53.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=240 avail_mem=53.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=224 avail_mem=53.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=208 avail_mem=53.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=192 avail_mem=53.50 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.13it/s]Capturing num tokens (num_tokens=192 avail_mem=53.50 GB):  71%|███████   | 41/58 [00:01<00:00, 40.21it/s]Capturing num tokens (num_tokens=176 avail_mem=53.49 GB):  71%|███████   | 41/58 [00:01<00:00, 40.21it/s]Capturing num tokens (num_tokens=160 avail_mem=53.49 GB):  71%|███████   | 41/58 [00:01<00:00, 40.21it/s]Capturing num tokens (num_tokens=144 avail_mem=53.48 GB):  71%|███████   | 41/58 [00:01<00:00, 40.21it/s]Capturing num tokens (num_tokens=128 avail_mem=53.48 GB):  71%|███████   | 41/58 [00:01<00:00, 40.21it/s]

    Capturing num tokens (num_tokens=112 avail_mem=53.48 GB):  71%|███████   | 41/58 [00:01<00:00, 40.21it/s]Capturing num tokens (num_tokens=112 avail_mem=53.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=96 avail_mem=53.48 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.31it/s] Capturing num tokens (num_tokens=80 avail_mem=53.47 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=64 avail_mem=53.47 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=48 avail_mem=53.47 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=32 avail_mem=53.42 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.31it/s]Capturing num tokens (num_tokens=32 avail_mem=53.42 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=28 avail_mem=53.41 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=24 avail_mem=53.41 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=20 avail_mem=53.40 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=16 avail_mem=53.40 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.54it/s]

    Capturing num tokens (num_tokens=12 avail_mem=53.40 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.54it/s]Capturing num tokens (num_tokens=12 avail_mem=53.40 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=8 avail_mem=53.40 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.84it/s] Capturing num tokens (num_tokens=4 avail_mem=53.39 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.84it/s]Capturing num tokens (num_tokens=4 avail_mem=53.39 GB): 100%|██████████| 58/58 [00:01<00:00, 31.51it/s]


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
    Generated text:  Tanyae and I am a digital marketing and influencer. I have over a decade of experience in SEO, PPC, and content marketing, and I have also worked with influencer marketing. I have a background in marketing and business, and I've always been passionate about helping businesses grow and thrive. My aim is to help my clients find the best strategies for their specific needs and to provide them with valuable insights and information. I'm also a strong believer in the importance of authenticity and transparency in my work, and I strive to make my clients feel supported and heard. I believe that every client is unique, and I take the time
    ===============================
    Prompt: The president of the United States is
    Generated text:  250 cm tall. His tallest competitor is 6 feet 8 inches tall. If it takes 2 seconds to walk the full 1 meter (100 cm), how long will it take for the president to walk the height of his competitor?
    To determine how long it will take for the president to walk the height of his tallest competitor, we need to follow these steps:
    
    1. Convert the height of the competitor from feet and inches to just centimeters.
    2. Calculate the distance the president needs to walk to cover the height of the competitor.
    3. Determine how long it will take for the president to walk
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. Lyon C. Nice D. Montmartre
    A: Paris
    B: Lyon
    C: Nice
    D: Montmartre
    
    To determine the capital of France, let's analyze the options provided:
    
    A. Paris
    B. Lyon
    C. Nice
    D. Montmartre
    
    We need to identify which of these is the capital of France. The capital of France is typically the largest city or city-state that serves as the main administrative center of the country.
    
    - Paris is the largest city in France and the capital of France.
    - Lyon is the capital of France.
    - Nice is
    ===============================
    Prompt: The future of AI is
    Generated text:  better than it is today.
    
    Q: The future of AI is better than it is today. What does that suggest about AI?
    
    Pick your answer from:
     --good
     --bad
     --incomprehensible
     --neutral
    To determine the correct answer, I will analyze the given statement and compare it with the options provided:
    
    1. The statement "The future of AI is better than it is today" implies that the future holds the potential for advancements in AI that were not previously possible.
    
    2. This suggests that the future holds more opportunities and capabilities for AI.
    
    3. The options provided do not directly address the positive aspects of the


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


    Generated text:  [Name] and I am a [job title] at [company name]. I am passionate about [reason for being at the company]. I am always looking for ways to [what I enjoy doing]. I am [age] years old. I have [number of years of experience] years of experience in [job title]. I am [gender] and I am [race]. I am [country]. I am [language]. I am [interests and hobbies]. I am [any other relevant information]. I am [any other relevant information]. I am [any other relevant information]. I am [any other relevant information]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and vibrant French culture. 
    
    (Note: The statement should be clear and concise, using appropriate terminology and avoiding any potentially sensitive or controversial content.) 
    
    The capital of France is Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and vibrant French culture. 
    
    (Note: The statement should be clear and concise, using appropriate terminology and avoiding any potentially sensitive or controversial content.) 
    
    The capital of France is Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and vibrant French culture. 
    
    (Note: The statement should be
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends in AI that are expected to shape the future:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This will lead to the development of new jobs, but it will also create new opportunities for people to work in areas such as data analysis, software development, and robotics.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs.
    


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
    Generated text:  [Name] and I'm a [role or profession] who has always been passionate about [your profession]. [Your profession] has taught me a lot and has shown me how to [something]. I'm confident that I can make a positive impact on my community and [reason why]. I'm a [value, like "outstanding, exceptional, or helpful"].
    My name is [Name], I'm a [role or profession], and I'm [your profession] who has always been passionate about [your profession]. [Your profession] has taught me a lot and has shown me how to [something]. I'm confident that
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Grande-Bretagne". It is the largest city in France, with a population of over 2 million people and is known for its famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Palace of Versailles. The city is also a UNESCO World Heritage site and is a popular tourist destination with a rich history, cultural heritage, and architecture. Paris is known for its cuisine, fashion, and art scene, and is a center of international commerce and finance. The city is also home to several museums, including the Louvre, which is one of the largest and most visited
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a wide range of different trends and innovations, each of which is likely to have a significant impact on the way we live, work, and interact with technology. Here are some of the key trends that are likely to shape the future of AI:
    
    1. Improved Machine Learning: With the growth of data and the development of more powerful computing power, machine learning is likely to become even more advanced. This means that AI systems will be able to learn from vast amounts of data, recognize patterns, and make decisions based on that data. This will lead to new applications of AI in fields such as healthcare, finance, and


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

    Age

    ]

     year

     old

    ,

     [

    Occup

    ation

    /

    Position

    ]

     in

     my

     [

    Industry

    /

    Company

    ].

     I

    've

     always

     had

     a

     love

     for

     learning

     and

     discovering

     new

     things

    ,

     and

     have

     always

     been

     passionate

     about

     traveling

     the

     world

     and

     connecting

     with

     different

     cultures

    .

     I

    've

     been

     in

     constant

     motion

    ,

     from

     working

     in

     my

     office

     to

     traveling

     to

     meet

     people

     and

     experience

     different

     cultures

    .

     I

    'm

     always

     looking

     for

     new

     opportunities

     to

     grow

     and

     learn

    ,

     and

     I

    'm

     grateful

     for

     the

     connections

     I

    've

     made

     along

     the

     way

    .

     I

     love

     being

     in

     new

     places

    ,

     trying

     new

     things

    ,

     and

     making

     new

     friends

    .

     My

     style

     is

     simple

    ,

     but

     kind

     and

     honest

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

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

     the

     Arc

     de

     Tri

    omp

    he

    .

     Additionally

    ,

     Paris

     is

     a

     popular

     tourist

     destination

     and

     home

     to

     numerous

     museums

    ,

     art

     galleries

    ,

     and

     theaters

    .

     It

     is

     also

     a

     significant

     financial

     and

     cultural

     center

     in

     Europe

    .

     
    


    The

     city

     of

     Paris

    ,

     also

     known

     as

     "

    La

     Ville

     Bl

    anche

    "

     (

    White

     City

    ),

     is

     a

     vibrant

    ,

     diverse

    ,

     and

     historic

     met

    ropolis

     that

     is

     home

     to

     

    2

    .

    8

     million

     people

     and

     a

     significant

     presence

     in

     French

     culture

    .

     The

     city

     is

     known

     for

     its

     art

    ,

     literature

    ,

     music

    ,

     and

     fashion

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     uncertain

     and

     unpredictable

    ,

     but

     there

     are

     a

     few

     possible

     trends

     that

     are

     currently

     being

     explored

     and

     are

     seen

     as

     promising

     areas

     for

     innovation

    .

     Here

     are

     some

     of

     the

     potential

     areas

     of

     development

     in

     AI

    :
    


    1

    .

     Natural

     Language

     Processing

     (

    N

    LP

    ):

     N

    LP

     is

     the

     ability

     of

     AI

     to

     understand

     and

     interpret

     human

     language

    .

     This

     area

     of

     AI

     is

     currently

     being

     explored

     by

     researchers

     at

     companies

     like

     Amazon

    ,

     Google

    ,

     and

     Microsoft

    ,

     and

     it

     has

     the

     potential

     to

     revolution

    ize

     how

     we

     communicate

     and

     interact

     with

     each

     other

    .
    


    2

    .

     Machine

     Learning

     and

     Deep

     Learning

    :

     Machine

     learning

     is

     a

     type

     of

     AI

     that

     involves

     developing

     algorithms

     that

     can

     learn

     from

     data

     and

     make

     predictions

     or

     decisions

    .

    



```python
llm.shutdown()
```
