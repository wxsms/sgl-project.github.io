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
    [2026-04-27 03:52:07] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.74it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.73it/s]


    2026-04-27 03:52:12,251 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-27 03:52:12] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:46,  5.02s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.89it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.30it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.80it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.80it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.80it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.80it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.80it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.80it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.80it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.80it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:02, 13.80it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:02, 13.80it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:02, 13.80it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 21.33it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 21.33it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 21.33it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 21.33it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 21.33it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 21.33it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 21.33it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 21.33it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 21.33it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 21.33it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 29.00it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.12it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:04, 12.73it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:04, 12.73it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   3%|▎         | 2/58 [00:00<00:04, 12.73it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.76 GB):   7%|▋         | 4/58 [00:00<00:03, 13.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.75 GB):   7%|▋         | 4/58 [00:00<00:03, 13.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.26 GB):   7%|▋         | 4/58 [00:00<00:03, 13.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.25 GB):   7%|▋         | 4/58 [00:00<00:03, 13.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.25 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.70it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.70it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.70it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.03it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.03it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.03it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.03it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 24.03it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  28%|██▊       | 16/58 [00:00<00:01, 30.67it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.67it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.67it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.54it/s]

    Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 38.54it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 40.63it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.63it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.63it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.63it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:01<00:00, 40.63it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.06it/s]

    Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.06it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 44.00it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 44.00it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  71%|███████   | 41/58 [00:01<00:00, 44.00it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 44.00it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 44.00it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  71%|███████   | 41/58 [00:01<00:00, 44.00it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.71it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.45it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.45it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.45it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.45it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.45it/s]

    Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.45it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.86it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.86it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 36.25it/s]


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
    Generated text:  Alex and I live in the country of East Asia. I am going to be an AI assistant for the government of the country. I can help people with information and advice.
    
    What did you learn about AI in the country of East Asia?
    I believe you are not asking about AI in the country of East Asia. It seems to be some kind of typo or confusion, so I will respond based on my understanding of AI.
    
    AI is a broad field that encompasses a wide range of technology and techniques, including but not limited to machine learning, natural language processing, computer vision, robotics, and others. It is an interdisciplinary field that involves the
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the government of the United States. As the head of the government, the president has the power to make decisions on important issues. The president also has to be trustworthy, fair, and honest, to get his or her people to trust him or her.
    The president is elected by the people to serve a term of four years. The president is a prime candidate for election in every state in the United States, and the people choose the person they want to serve as the president. They are responsible for overseeing the country and making important decisions in the government.
    In the United States, the president is seen as the leader
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris. Its history goes back to the 12th century, it is the most populous city in the world. In 1830, it became the capital of the French Empire. In 1870, it was changed to the capital of the French Republic. In 1962, the city was renamed to Paris, after the 1962 Olympic Games. Paris is known for its historical buildings, such as the Eiffel Tower, the Louvre Museum, and the Arc de Triomphe. It is also famous for its food, music, and fashion. Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  still in its early stages, but the day is approaching when we'll be able to train a computer to do tasks that require human intelligence. This point is coming closer with the latest advancement in artificial intelligence. It's now possible to train a computer to recognize and manipulate a wide variety of emotions and behaviors. This is a significant advancement that can have a wide range of applications, from developing AI assistants to improving customer service. The potential for AI to help humans in a wide variety of ways has created a need for a team of experts to develop these technologies.
    One of the biggest challenges in developing AI is the complex task of understanding and manipulating emotions


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" and the "City of Light". It is the largest city in France and the third-largest city in the world by population. The city is located on the Seine River and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is known for its rich history, art, and culture, and is a popular tourist destination for visitors from around the world. The city is also home to many important institutions and organizations, including the French Academy of Sciences and the French National Library. Paris is a vibrant and dynamic city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some potential trends that are likely to shape the future of AI:
    
    1. Increased automation and robotics: As AI technology continues to improve, we are likely to see an increase in automation and robotics in various industries. This could lead to the creation of more efficient and productive machines that can perform tasks that were previously done by humans.
    
    2. AI ethics and privacy concerns: As AI technology becomes more advanced, there will be increasing concerns about the ethical implications of AI. This includes issues such as bias, privacy
    


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
    Generated text:  [Name], and I'm [Age] years old. I currently live in [City/Town] and I'm a [occupation] at heart. I'm a [occupation] who has always been interested in learning and exploring the world around me. I have a love for [activity/interest], and I'm always on the lookout for new experiences. I'm a [occupation] who loves to make people smile and I'm always up for a challenge. I'm a [occupation] who is always seeking out new opportunities and I'm always open to trying new things. I love spending time with my family and friends, and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its rich history, vibrant culture, and iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. 
    
    In Paris, you can enjoy a diverse range of food and drink options, including traditional French cuisine, international cuisine, and speciality dishes. The city is also famous for its museums, theaters, and charming neighborhoods such as Montmartre, the Latin Quarter, and Marais. Paris is a cultural and artistic destination, known for its museums, theaters, and dance performances. 
    
    Paris is a modern and innovative city, with a strong emphasis on science and technology. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several trends, including:
    
    1. Increased automation and robotics: As AI technology continues to advance, we may see more widespread adoption of automation and robotics in various industries, including manufacturing, transportation, and healthcare. This could lead to increased efficiency and productivity, but could also result in job displacement for some workers.
    
    2. Increased use of AI in self-driving vehicles: With the ongoing rollout of self-driving cars, AI will become even more important in transportation, with autonomous vehicles becoming a major focus of research and development.
    
    3. Increased use of AI in medicine: AI can help doctors diagnose diseases, predict patient outcomes, and


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

    __.

     I

     am

     a

    /an

     __

    __.

     I

     am

     an

     __

    __.

     I

     have

     a

    /an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    __.

     I

     am

     an

     __

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     city

     located

     in

     the

     Se

    ine

    -Mar

    itime

     region

     and

     known

     for

     its

     stunning

     architecture

    ,

     rich

     history

    ,

     and

     diverse

     culture

    .

     It

     is

     the

     oldest

     capital

     city

     in

     the

     world

    ,

     dating

     back

     to

     the

     

    6

    th

     century

    ,

     and

     is

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

    .

     Paris

     is

     also

     known

     for

     its

     culinary

     traditions

    ,

     fashion

    ,

     and

     music

     scene

    ,

     and

     is

     home

     to

     many

     world

    -ren

    owned

     museums

    ,

     art

     galleries

    ,

     and

     monuments

    .

     It

     is

     a

     city

     that

     is

     consistently

     referred

     to

     as

     "

    the

     city

     of

     love

    "

     due

     to

     its

     romantic

     attractions

     and

     romantic

     cafes

    ,

     and

     is

     a

     popular

     destination

     for

     tourists

    ,

     artists

    ,

     and

     locals

     alike

    .

     Paris

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     shaped

     by

     a

     multitude

     of

     factors

    ,

     including

     advances

     in

     technology

    ,

     increasing

     complexity

    ,

     and

     growing

     complexity

    .

     Some

     of

     the

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

     Automation

    :

     With

     the

     development

     of

     advanced

     machine

     learning

     algorithms

    ,

     we

     can

     expect

     to

     see

     more

     automation

     in

     the

     workplace

    ,

     where

     machines

     will

     be

     able

     to

     perform

     tasks

     that

     were

     previously

     done

     by

     humans

    .
    


    2

    .

     Increased

     Eth

    ical

     Consider

    ations

    :

     As

     AI

     continues

     to

     develop

    ,

     it

     will

     become

     increasingly

     important

     to

     consider

     the

     ethical

     implications

     of

     its

     development

    ,

     such

     as

     privacy

    ,

     bias

    ,

     and

     accountability

    .
    


    3

    .

     Improved

     Privacy

    :

     With

     the

     increasing

     use

     of

     AI

     in

     everyday

     life

    ,

     there

     will

     be

     greater

    



```python
llm.shutdown()
```
