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
    [2026-04-23 23:53:54] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.98it/s]


    2026-04-23 23:53:59,747 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-23 23:53:59] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.83it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.63it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.63it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.63it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.63it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.63it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.63it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.63it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.63it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.63it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:03<00:08,  5.63it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:03<00:08,  5.63it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:03<00:02, 13.56it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:03<00:02, 13.56it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:03<00:02, 13.56it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:03<00:02, 13.56it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 13.56it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 13.56it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 13.56it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 13.56it/s]

    Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 13.56it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 13.56it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 21.72it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 30.81it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 30.81it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 30.81it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 30.81it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 30.81it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 30.81it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 30.81it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:03<00:00, 30.81it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:03<00:00, 30.81it/s]

    Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:03<00:00, 30.81it/s] Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:03<00:00, 30.81it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 41.75it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 41.75it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 41.75it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 41.75it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 41.75it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 41.75it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 41.75it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 41.75it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 41.75it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 41.75it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 41.75it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=76.80 GB):   2%|▏         | 1/58 [00:00<00:12,  4.41it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   2%|▏         | 1/58 [00:00<00:12,  4.41it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:08,  6.43it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:08,  6.43it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:08,  6.43it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:08,  6.43it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:03, 13.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:03, 13.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:03, 13.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:03, 13.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):   9%|▊         | 5/58 [00:00<00:03, 13.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.12it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.19it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 33.19it/s]Capturing num tokens (num_tokens=832 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.79it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.79it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.79it/s]Capturing num tokens (num_tokens=640 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.79it/s]Capturing num tokens (num_tokens=576 avail_mem=76.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.79it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 37.79it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  50%|█████     | 29/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:01<00:00, 41.29it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  50%|█████     | 29/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  50%|█████     | 29/58 [00:01<00:00, 41.29it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=240 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:01<00:00, 44.29it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.79it/s]

    Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  69%|██████▉   | 40/58 [00:01<00:00, 45.79it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.23it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  78%|███████▊  | 45/58 [00:01<00:00, 44.23it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.79it/s]

    Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.79it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.13it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.13it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.13it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.13it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 35.21it/s]


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
    Generated text:  William and I am from the United States. I have a post-graduate degree in English literature and am a member of the Chicago Board of Trade. I enjoy teaching English as a second language, sharing my knowledge of English literature, and running a blog called The English Language. My own blog is called The English Language Blog.
    My blog is on a regular basis updated with fresh content and insights from my own personal experiences, as well as from other bloggers. The English Language Blog also includes posts from other blogs, so you can find helpful links to other English-language blogs.
    Here are a few of the things that I do on the blog:
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for a third term. The current president has served for 8 years and is 40 years old. The new president is predicted to be 25 years younger than the current president but 50 years older than the current president. How many years old is the new president?
    To determine the age of the new president, we need to follow these steps:
    
    1. Identify the current age of the president.
    2. Calculate the predicted age of the new president based on the given information.
    3. Add the predicted age to the current age of the president to find the new president's age.
    
    First, we know the current
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, it has a population of 2,543,841. Paris, also known as “Père Lachaise” is the capital of the country. It is located in the east of France and was founded by the French Kingdom of France. The capital of France has a population of 2,543,841 people, making it the most populous country in Europe. The city is also the cultural and economic center of the country and plays a significant role in the development of the nation.
    The capital of France is Paris and it is also known as Père Lachaise. It has
    ===============================
    Prompt: The future of AI is
    Generated text:  always changing, and as AI technology continues to advance, its applications will continue to expand. It is expected that by the end of the decade, AI will play a critical role in many industries, from healthcare to transportation, and from finance to entertainment. However, with the increasing demand for advanced AI technology, there are also concerns about the ethical implications of using AI to solve problems.
    One of the most common ethical concerns with AI is that it can lead to job displacement. As AI becomes more prevalent in our lives, it is possible that some jobs will become obsolete or will require more specialized skills to perform. This can have a significant impact on


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [Job Title] at [Company Name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [Job Title] at [Company Name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [Job Title] at [Company Name], and I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a popular tourist destination and is home to many world-renowned museums, such as the Louvre and the Musée d'Orsay. The city is also known for its cuisine, with its famous dishes such as croissants,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and preferences. This could lead to more natural and intuitive interactions between humans and machines.
    
    2. Enhanced machine learning capabilities: AI is likely to become more powerful and capable, allowing machines to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective decision-making in various industries.
    
    3. Increased focus on ethical and responsible AI: As AI becomes more integrated with human intelligence, there
    


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
    Generated text:  [Name], and I am a [type of character], specifically [type of character]. I have always had a passion for [activity or hobby], and I believe that [motivation or interest] is the key to success. I am always eager to learn new things and adapt to different environments, so I am confident that I have the potential to make a positive impact in the world. I believe that I have a lot to offer, and I am always looking for the best ways to contribute to the world. Thank you for having me. Let me know if you'd like me to elaborate on any of this. [Name] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Does this next sentence follow, given the above sentence? "Paris is in the United States."?
    
    a). yes;
    b). it is not possible to tell;
    c). no;
    c). no;
    
    The sentence "Paris is in the United States" is not supported by the given statement "The capital of France is Paris." While Paris is located in France, it is not in the United States. France is an independent country, while the United States is a country in the Americas. Therefore, the statement about Paris being in the United States is false. 
    
    b). it is not possible to tell is incorrect because we have
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be exciting and innovative, driven by the rapid advancements in technology. Here are some possible trends that we can expect to see in the near future:
    
    1. Increased use of AI in healthcare: AI is increasingly being used in the healthcare industry to diagnose and treat diseases. For example, AI-powered medical imaging systems can analyze medical images more accurately and quickly than human doctors. Additionally, AI-powered chatbots can provide 24/7 patient support and can handle more complex healthcare scenarios.
    
    2. Increased use of AI in manufacturing: AI is being used to optimize production processes, reduce costs, and improve quality. For example, AI can


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

    ].

     I

     am

     [

    Your

     Age

    ]

     years

     old

     and

     currently

     reside

     in

     [

    Your

     City

    /T

    own

    ].

     I

     have

     always

     been

     interested

     in

     the

     arts

     and

     have

     been

     playing

     the

     violin

     since

     I

     was

     [

    Your

     Age

    ].

     I

     enjoy

     painting

     and

     creating

     my

     own

     art

     in

     a

     variety

     of

     mediums

    ,

     and

     I

     am

     always

     looking

     for

     new

     experiences

     and

     opportunities

     to

     explore

     my

     creativity

    .

     My

     goal

     is

     to

     continue

     to

     grow

     and

     develop

     as

     a

     person

     and

     to

     be

     a

     better

     version

     of

     myself

     every

     day

    .

     Thank

     you

     for

     asking

     about

     me

     and

     for

     the

     opportunity

     to

     introduce

     myself

    .

     I

     look

     forward

     to

     hearing

     more

     about

     you

    !

     If

     you

     have

     any

     questions

     or

     need

     further

     information

    ,

    
    
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

     Notre

     Dame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     also

     famous

     for

     its

     rich

     cultural

     heritage

     and

     cosm

    opolitan

     atmosphere

    .

     France

    's

     capital

     is

     situated

     on

     the

     Lo

    ire

     River

     and

     is

     home

     to

     over

     

    3

     million

     people

    .

     The

     capital

     is

     a

     major

     transportation

     hub

    ,

     with

     numerous

     train

     stations

     and

     fer

    ries

     connecting

     the

     city

     to

     nearby

     cities

     and

     the

     continent

    .

     The

     city

     is

     also

     known

     for

     its

     international

     restaurants

    ,

     cafes

    ,

     and

     bars

    ,

     with

     a

     vibrant

     nightlife

     and

     cultural

     scene

    .

     Paris

     is

     home

     to

     many

     renowned

     museums

    ,

     including

     the

     Lou

    vre

    ,

     the

     Tate

     Modern

    ,

     and

     the

     National

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

     and

     there

     are

     a

     number

     of

     potential

     trends

     we

     can

     expect

     to

     see

     in

     the

     years

     ahead

    .

     Here

     are

     some

     of

     the

     key

     areas

    :
    


    1

    .

     More

     autonomous

     and

     adaptive

     systems

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

     more

     autonomous

     and

     adaptive

     systems

     that

     can

     handle

     a

     wide

     range

     of

     tasks

     more

     effectively

     than

     human

     beings

    .

     For

     example

    ,

     machines

     that

     can

     learn

     from

     experience

    ,

     adapt

     to

     new

     tasks

     and

     situations

    ,

     and

     make

     decisions

     in

     real

    -time

     will

     be

     more

     common

    .
    


    2

    .

     Increased

     focus

     on

     ethical

     and

     societal

     implications

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     there

     will

     be

     increased

     focus

     on

     how

     it

     affects

     human

     society

     and

     the

     broader

    



```python
llm.shutdown()
```
