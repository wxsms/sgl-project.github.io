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
    [2026-04-20 21:48:58] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.18it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.17it/s]


    2026-04-20 21:49:02,737 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-20 21:49:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:31,  2.65s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.89it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=1280):  17%|█▋        | 10/58 [00:02<00:08,  5.81it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:02<00:02, 13.96it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:02<00:02, 13.96it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:02<00:02, 13.96it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:02<00:02, 13.96it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:03<00:02, 13.96it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:03<00:02, 13.96it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 22.17it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 22.17it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 22.17it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 22.17it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 22.17it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 22.17it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 22.17it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 22.17it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:03<00:01, 22.17it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:03<00:01, 22.17it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:03<00:01, 22.17it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 32.35it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:03<00:00, 32.35it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 42.65it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 42.65it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 42.65it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 42.65it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 42.65it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 42.65it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 42.65it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 42.65it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 42.65it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 42.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.16it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=61.77 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=61.75 GB):   3%|▎         | 2/58 [00:00<00:03, 18.41it/s]Capturing num tokens (num_tokens=7168 avail_mem=61.74 GB):   3%|▎         | 2/58 [00:00<00:03, 18.41it/s]Capturing num tokens (num_tokens=6656 avail_mem=61.74 GB):   3%|▎         | 2/58 [00:00<00:03, 18.41it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=61.74 GB):   7%|▋         | 4/58 [00:00<00:02, 18.36it/s]Capturing num tokens (num_tokens=6144 avail_mem=61.74 GB):   7%|▋         | 4/58 [00:00<00:02, 18.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=60.37 GB):   7%|▋         | 4/58 [00:00<00:02, 18.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=60.37 GB):   7%|▋         | 4/58 [00:00<00:02, 18.36it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=60.37 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=60.37 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=60.37 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=3840 avail_mem=60.36 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=60.36 GB):  12%|█▏        | 7/58 [00:00<00:03, 15.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=60.36 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=3328 avail_mem=60.35 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=3072 avail_mem=60.35 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=2816 avail_mem=60.35 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=60.34 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.65it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=60.34 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=60.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=60.34 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=60.33 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=1536 avail_mem=60.33 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=60.33 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.31 GB):  28%|██▊       | 16/58 [00:00<00:01, 28.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=60.31 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.55it/s]Capturing num tokens (num_tokens=960 avail_mem=60.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.55it/s] Capturing num tokens (num_tokens=896 avail_mem=60.32 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.55it/s]Capturing num tokens (num_tokens=832 avail_mem=60.31 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.55it/s]Capturing num tokens (num_tokens=768 avail_mem=60.31 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.55it/s]

    Capturing num tokens (num_tokens=704 avail_mem=60.31 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.55it/s]Capturing num tokens (num_tokens=704 avail_mem=60.31 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=640 avail_mem=60.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=576 avail_mem=60.30 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=512 avail_mem=60.29 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=480 avail_mem=60.31 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=448 avail_mem=60.31 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.47it/s]Capturing num tokens (num_tokens=448 avail_mem=60.31 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=416 avail_mem=60.30 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=384 avail_mem=60.30 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=352 avail_mem=60.30 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=320 avail_mem=60.29 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.79it/s]

    Capturing num tokens (num_tokens=288 avail_mem=60.29 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.79it/s]Capturing num tokens (num_tokens=288 avail_mem=60.29 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=256 avail_mem=60.29 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=240 avail_mem=60.28 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=224 avail_mem=60.28 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=208 avail_mem=60.28 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=192 avail_mem=60.28 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.89it/s]Capturing num tokens (num_tokens=192 avail_mem=60.28 GB):  71%|███████   | 41/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=176 avail_mem=60.27 GB):  71%|███████   | 41/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=160 avail_mem=60.27 GB):  71%|███████   | 41/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=144 avail_mem=60.26 GB):  71%|███████   | 41/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=128 avail_mem=60.26 GB):  71%|███████   | 41/58 [00:01<00:00, 43.74it/s]

    Capturing num tokens (num_tokens=112 avail_mem=60.26 GB):  71%|███████   | 41/58 [00:01<00:00, 43.74it/s]Capturing num tokens (num_tokens=112 avail_mem=60.26 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=96 avail_mem=60.26 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.01it/s] Capturing num tokens (num_tokens=80 avail_mem=60.25 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=64 avail_mem=60.25 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=48 avail_mem=60.25 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=32 avail_mem=60.25 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.01it/s]Capturing num tokens (num_tokens=32 avail_mem=60.25 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=28 avail_mem=60.24 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=24 avail_mem=60.24 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=20 avail_mem=60.23 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=16 avail_mem=60.23 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.42it/s]

    Capturing num tokens (num_tokens=12 avail_mem=60.23 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.42it/s]Capturing num tokens (num_tokens=12 avail_mem=60.23 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.30it/s]Capturing num tokens (num_tokens=8 avail_mem=60.22 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.30it/s] Capturing num tokens (num_tokens=4 avail_mem=60.22 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.30it/s]Capturing num tokens (num_tokens=4 avail_mem=60.22 GB): 100%|██████████| 58/58 [00:01<00:00, 37.24it/s]


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
    Generated text:  Jia and I am a college student. I was found guilty of a rather heinous crime, which I have confessed to. The punishment I have been ordered to undergo is to have my social security card temporarily suspended for three days. This is a very severe punishment. 
    
    I have always wanted to travel and visit my family members, and I am very grateful to my parents. I also have a strong desire to travel and visit my family members, and I am very grateful to my parents. However, due to the severity of the punishment, I am unable to travel. 
    
    I have always been trying to save my parents' life,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the U.S. Congress and has the power to veto the bill passed by the U.S. Senate. The president also has the power to appoint the members of the U.S. Supreme Court. The president is also the head of the executive branch of the United States government. 
    
    While the president is a member of the U.S. Congress, they do not have the power to impeach the president. However, the president can be impeached for serious crimes and misdeeds, such as treason, forgery, bribery, corruption, and abuse of power.
    
    The United States Constitution grants the president the powers of both the
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and the longest street in France is the Seine River. The Roman era of the capital of France is recorded in its name, "París."
    
    What is the capital of France? The capital of France is Paris. It is known for its history, art, and culture, with several landmarks and attractions that make it one of the most visited cities in the world. Paris is famous for its stunning architecture, such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral, as well as its world-renowned cuisine and music. The city's rich history, including its Roman origins and its long
    ===============================
    Prompt: The future of AI is
    Generated text:  in human hands: researchers at MIT have created a new system called Mira that can predict future hiring trends in the tech industry. Mira is an AI system that uses machine learning to predict which companies will be hiring people in the future and what skills they will need.
    The company was founded by a group of professors at MIT and a company called Clearlake, and it aims to help companies make more informed hiring decisions by providing them with insights into future demand for specific skills. The system is built on a combination of data science, machine learning, and natural language processing.
    The researchers behind Mira say that their system can accurately predict which companies


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill] with [Number] years of experience in [Field]. I'm passionate about [What I Love to Do]. I'm always looking for new challenges and opportunities to grow and learn. I'm a [What I Like to Do] and I enjoy [What I Like to Do]. I'm a [What I Like to Do] and I enjoy [What I Like to Do]. I'm a [What I Like to Do] and I enjoy [What I Like to Do]. I'm a [What I Like to Do]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its fashion industry, art scene, and cuisine. Paris is a vibrant and cosmopolitan city with a diverse population and a rich cultural heritage. It is the largest city in France by population and is a major economic and political center in Europe. The city is home to many world-renowned museums, theaters, and art galleries, and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that could be expected in the AI field:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This could include issues such as bias, transparency, and accountability. AI developers will need to be more mindful of the potential impact of their technology on society and work to ensure that it is used in a responsible and ethical manner.
    
    2. Greater integration with other technologies: AI is already being integrated into
    


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
    Generated text:  [insert name], and I'm a [insert occupation] who is always looking for fresh ideas and creative solutions to challenges. I'm always up for a challenge and am always ready to collaborate with others. I have a passion for creating innovative solutions that solve problems and bring new perspectives to the table. I'm excited to continue exploring my creativity and help others grow by offering my ideas and resources. Thank you for taking the time to meet me. [Insert name]  
    Describe a scene from your current project. As an AI language model, I don't have a physical presence and therefore cannot provide a scene from a current project. However,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "La Paroisse." Paris is the city of light and culture and is the largest city in Europe, home to around 20 million people. It's known for its architecture, museums, and romantic ambiance, and is a popular tourist destination. The city has a rich history dating back over 2, 000 years, and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is known for its fashion industry, fine dining, and lively nightlife, and is an important cultural and political center of Europe. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and rapidly evolving. Here are some possible trends in the next few years:
    
    1. AI will continue to become more accurate and sophisticated. We'll see more machines that can make better decisions, understand complex problems and communicate with humans in new ways.
    
    2. AI will become even more ubiquitous in our lives. We'll see more AI-powered tools and applications in everything from healthcare to finance to transportation.
    
    3. AI will continue to play a more significant role in shaping the future of work. As automation and artificial intelligence become more prevalent, we'll see more jobs being replaced by machines, but also more opportunities being created.
    
    4. AI will


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

     an

     [

    Occup

    ation

    /

    Position

    ]

     [

    Occup

    ation

    /

    Position

    ].

     I

     have

     a

     passion

     for

     [

    What

     interests

     me

    /

    What

     do

     I

     like

     to

     do

     in

     my

     free

     time

    ]

     and

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

    .

     I

    'm

     a

     [

    how

     do

     you

     show

     your

     personality

    ,

     interests

    ,

     etc

    .

    ]?

     I

    'm

     always

     willing

     to

     learn

     and

     adapt

    ,

     and

     I

    'm

     eager

     to

     share

     my

     experiences

     and

     knowledge

     with

     others

    .

     Looking

     forward

     to

     making

     new

     friends

     and

     experiencing

     new

     things

    !

     

    😊

    ✨

    🌟

    
    


    Hey

     there

    ,

     [

    Name

    ]

    !

     It

    's

     nice

     to

     meet

     you

    .

     How

    's

     it

     going

    ?

     I

    'm

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     city

     known

     for

     its

     romantic

     architecture

    ,

     vibrant

     culture

    ,

     and

     annual

     La

     F

    ête

     de

     la

     Cro

    ix

     de

     France

    ,

     a

     French

     national

     holiday

    .

     Paris

     is

     also

     known

     for

     its

     exceptional

     food

    ,

     world

    -ren

    owned

     fashion

    ,

     and

     vibrant

     nightlife

    .

     Additionally

    ,

     the

     city

     is

     a

     major

     center

     for

     international

     politics

     and

     economics

    ,

     with

     the

     E

    iff

    el

     Tower

     as

     one

     of

     the

     world

    's

     most

     recognizable

     landmarks

    .

     Paris

    ,

     known

     as

     the

     "

    City

     of

     Light

    ,"

     has

     a

     rich

     cultural

     heritage

    ,

     including

     art

     and

     literature

    ,

     music

     and

     theater

    ,

     and

     a

     lively

     nightlife

    .

     The

     city

    's

     climate

     is

     temper

    ate

     with

     two

     distinct

     seasons

    ,

     and

     it

     is

     the

     largest

     metropolitan

     area

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     number

     of

     technological

     and

     organizational

     changes

     that

     will

     shape

     the

     way

     in

     which

     AI

     is

     developed

    ,

     deployed

    ,

     and

     integrated

     into

     our

     daily

     lives

    .

     Some

     potential

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     use

     of

     AI

     in

     healthcare

    :

     AI

     will

     be

     used

     to

     improve

     the

     accuracy

     and

     speed

     of

     medical

     diagnosis

    ,

     develop

     new

     treatments

    ,

     and

     predict

     patient

     outcomes

    .

     This

     will

     require

     the

     development

     of

     new

     algorithms

     and

     data

     sets

     to

     provide

     high

    -quality

     medical

     images

     and

     data

     for

     training

     AI

     models

    .
    


    2

    .

     Expansion

     of

     AI

     in

     areas

     such

     as

     finance

    ,

     finance

    ,

     and

     retail

    :

     AI

     will

     be

     used

     to

     automate

     customer

     service

    ,

     improve

     fraud

     detection

    ,

     and

     provide

     more

    



```python
llm.shutdown()
```
