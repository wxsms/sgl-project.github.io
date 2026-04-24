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
    [2026-04-24 07:05:56] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.84it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.83it/s]


    2026-04-24 07:06:03,450 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-24 07:06:03] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.75it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.75it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.75it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.75it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.75it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:02<00:06,  6.75it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:02<00:06,  6.75it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.75it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.75it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.75it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:03<00:06,  6.75it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:03<00:02, 14.64it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:03<00:02, 14.64it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:03<00:02, 14.64it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:03<00:02, 14.64it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:03<00:02, 14.64it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:03<00:02, 14.64it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:03<00:02, 14.64it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:03<00:02, 14.64it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:03<00:02, 14.64it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:03<00:02, 14.64it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:03<00:01, 22.68it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:03<00:01, 22.68it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:03<00:01, 22.68it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:03<00:01, 22.68it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:03<00:01, 22.68it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:03<00:01, 22.68it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:03<00:01, 22.68it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:03<00:01, 22.68it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:03<00:01, 22.68it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:03<00:01, 22.68it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:03<00:01, 22.68it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 32.87it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:03<00:00, 32.87it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:03<00:00, 32.87it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 43.19it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 43.19it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 43.19it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 43.19it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 43.19it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 43.19it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 43.19it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 43.19it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.28 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.25 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.25 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.25 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.25 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.25 GB):   3%|▎         | 2/58 [00:00<00:03, 18.65it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.25 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.25 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.25 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.24 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.24 GB):   9%|▊         | 5/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.74it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.24 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.74it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.74it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.74it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.23 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.74it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.23 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.22 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.21 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.65it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.21 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.58it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.18 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.58it/s]Capturing num tokens (num_tokens=960 avail_mem=76.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.58it/s] Capturing num tokens (num_tokens=896 avail_mem=76.20 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.58it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.19 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.58it/s]Capturing num tokens (num_tokens=832 avail_mem=76.19 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=768 avail_mem=76.19 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=704 avail_mem=76.19 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=640 avail_mem=76.18 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=576 avail_mem=76.18 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=512 avail_mem=76.17 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.96it/s]Capturing num tokens (num_tokens=512 avail_mem=76.17 GB):  50%|█████     | 29/58 [00:00<00:00, 43.35it/s]Capturing num tokens (num_tokens=480 avail_mem=76.19 GB):  50%|█████     | 29/58 [00:00<00:00, 43.35it/s]Capturing num tokens (num_tokens=448 avail_mem=76.18 GB):  50%|█████     | 29/58 [00:00<00:00, 43.35it/s]Capturing num tokens (num_tokens=416 avail_mem=76.18 GB):  50%|█████     | 29/58 [00:00<00:00, 43.35it/s]Capturing num tokens (num_tokens=384 avail_mem=76.18 GB):  50%|█████     | 29/58 [00:00<00:00, 43.35it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.17 GB):  50%|█████     | 29/58 [00:00<00:00, 43.35it/s]Capturing num tokens (num_tokens=352 avail_mem=76.17 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.78it/s]Capturing num tokens (num_tokens=320 avail_mem=76.17 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.78it/s]Capturing num tokens (num_tokens=288 avail_mem=76.17 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.78it/s]Capturing num tokens (num_tokens=256 avail_mem=76.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.78it/s]Capturing num tokens (num_tokens=240 avail_mem=76.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.78it/s]Capturing num tokens (num_tokens=224 avail_mem=76.16 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.78it/s]Capturing num tokens (num_tokens=224 avail_mem=76.16 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=208 avail_mem=76.16 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=192 avail_mem=76.16 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=176 avail_mem=76.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.27it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.15 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=144 avail_mem=76.14 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.27it/s]Capturing num tokens (num_tokens=144 avail_mem=76.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=128 avail_mem=76.14 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.71it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.71it/s]

    Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.01it/s]Capturing num tokens (num_tokens=48 avail_mem=75.97 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.01it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.01it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.01it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.01it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 36.01it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.50it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.50it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.50it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.50it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 38.50it/s]

    Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 38.33it/s]


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
    Generated text:  Kuro and I'm a 16 year old girl. I have a lot of troubles. I used to be very happy and active in my childhood. However, I've always been in a relationship and my parents have been too strict with me, which caused me to feel lonely, depressed, anxious, and anxious. Is this normal? What can I do to change my situation? What will happen if I don't? I'm afraid I will not find a happy life because of these problems. I don't want to break up with my boyfriend, though he's not my boyfriend but I like him. I don't want to
    ===============================
    Prompt: The president of the United States is
    Generated text:  an official appointed by the President of the United States from among their nominees, and confirmed by the U.S. Senate. The President appoints federal judges, presidencies, ambassadors, and cabinet officers, and appoints federal judges to the Supreme Court of the United States. The President also appoints ambassadors to foreign nations, but has no power to veto or remove them.
    
    Based on the above text, what's the best answer to this question: are presidential appointments on the same scale as elected ones? Yes, presidential appointments on the same scale as elected ones. The text explicitly states that "the President appoints federal judges, presid
    ===============================
    Prompt: The capital of France is
    Generated text: 
    
      • Paris
      • Lyon
      • Tours
      • Marseille
    The capital of France is Paris. This is the most populous city in the European Union, and is the country's largest city. The city's population as of 2018 was 6.7 million (as of the 2020 census). Paris is the 2nd largest city in the world by population and the 1st largest by area.
    
    Paris is a major cultural, political, economic, and transportation centre in Europe and North-Western and Central Asia, and is a major global influence in politics, trade, industry
    ===============================
    Prompt: The future of AI is
    Generated text:  here
    
    In an age where the world is constantly changing and evolving, AI technology has been revolutionizing businesses and driving progress in many industries. But what exactly is AI? What is the future of AI? What does it mean for businesses and individuals? In this blog post, we will explore the future of AI, what it means for businesses and individuals, and the key developments that are shaping the future of AI. We will also examine the risks and challenges that come with AI and how businesses can stay ahead of the curve.
    
    AI is the technology that allows computers to learn from and make decisions based on data. In the past, AI was


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name], and I'm excited to meet you. I'm a [job title] at [company name],
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the third-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major cultural and economic center, hosting numerous world-renowned museums, theaters, and music venues. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is home to many important institutions and organizations, including the French Academy of Sciences and the French National Library. Paris is a vibrant and diverse city with a rich history and a strong sense
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to evolve and improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI systems become more advanced, there will be a greater emphasis on ensuring that they are used ethically and responsibly. This may involve developing new ethical guidelines and standards for AI systems, as well as increasing transparency and accountability in their development and deployment.
    
    2. Greater integration with human decision
    


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
    Generated text:  [Name], and I am [Age] years old. I come from [Country] and have been a [Occupation or Interest] for [Number] years. I'm always [Enjoying or Content] and I enjoy [Why]? I'm excited to meet you and discuss [What's on Your Mind?].
    
    Wow, [Name] sounds amazing! How did you end up in [Occupation or Interest] in your early twenties? Do you have any advice for people considering a similar career path? My name is [Name], and I am [Age] years old. I come from [Country] and have been a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the "City of Light."
    
    Key details:
    - City located in the western part of the country, bordering the English Channel
    - Eponym: The name Paris comes from the Latin word for light
    - Population: Approximately 2.1 million people as of 2021 (estimated)
    - Official language: French
    - Largest city by area: Paris
    - Famous landmarks: Eiffel Tower, Louvre Museum, Notre Dame Cathedral
    - Daylight saving time: 1 PM (Eastern Daylight Time) and 11 PM (Central Daylight Time)
    
    Geographical context:
    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by continued growth and development, with many different areas of research and application emerging. Some possible future trends in AI include:
    
    1. Autonomous vehicles: Self-driving cars and other self-driving vehicles are likely to become more common, with AI technologies being used to improve safety, efficiency, and convenience.
    
    2. Smart homes: AI technologies are already being used in smart homes to control appliances, automate tasks, and improve energy efficiency. As technology advances, we may see even more integration between the home and the devices we use, such as voice assistants and smart home devices.
    
    3. Predictive analytics: With the increasing amount of data


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

    insert

     name

    ]

     and

     I

    'm

     here

     to

     introduce

     myself

     to

     you

    .

     I

    'm

     [

    insert

     any

     relevant

     personal

     information

     or

     background

     information

    ].

     I

     have

     a

     wide

     range

     of

     interests

     and

     experiences

     that

     are

     unique

     to

     me

    ,

     but

     I

     also

     have

     a

     strong

     sense

     of

     empathy

     and

     a

     desire

     to

     help

     others

     in

     need

    .

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     always

     eager

     to

     contribute

     my

     skills

     and

     knowledge

     to

     help

     people

     in

     their

     time

     of

     need

    .

     Thank

     you

     for

     having

     me

    ,

     and

     I

     hope

     to

     hear

     from

     you

     soon

    .

     [

    insert

     any

     additional

     information

     that

     you

     want

     to

     include

    ].

     Good

    bye

    !

     [

    insert

     any

     closing

     remarks

     or

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    The

     statement

     can

     be

     varied

     depending

     on

     the

     context

    ,

     but

     for

     this

     example

    ,

     it

     would

     be

    :

     "

    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     country

    's

     capital

    ."

     
    


    For

     a

     more

     detailed

     and

     accurate

     statement

    ,

     a

     summary

     might

     be

    :

     "

    Paris

     is

     the

     largest

     city

     in

     France

    ,

     as

     well

     as

     the

     country

    's

     capital

    ,

     with

     a

     population

     of

     approximately

     

    2

    .

    1

     million

     people

    ."

     
    


    Which

     of

     these

     statements

     accurately

     summarizes

     the

     location

     and

     significance

     of

     Paris

     in

     France

    ?

     
    


    A

    )

     Statement

     

    1

    :

     "

    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     country

    's

     capital

    ."


    B

    )

     Statement

     

    2

    :

     "

    Paris

     is

     the

     country

    's

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     uncertain

    ,

     but

     some

     possible

     trends

     that

     are

     likely

     to

     shape

     its

     development

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

     With

     the

     rise

     of

     tele

    medicine

     and

     AI

    -driven

     diagnostic

     tools

    ,

     we

     may

     see

     more

     widespread

     adoption

     of

     AI

     in

     healthcare

    ,

     enabling

     better

     patient

     outcomes

     and

     reducing

     healthcare

     costs

    .
    


    2

    .

     AI

     integration

     into

     everyday

     life

    :

     AI

    -powered

     assistants

     like

     virtual

     assistants

    ,

     chat

    bots

    ,

     and

     drones

     may

     become

     more

     prevalent

    ,

     transforming

     our

     interactions

     with

     technology

     and

     our

     daily

     lives

    .
    


    3

    .

     AI

     integration

     into

     manufacturing

    :

     AI

     can

     improve

     production

     efficiency

    ,

     reduce

     waste

    ,

     and

     increase

     productivity

    ,

     making

     AI

     more

     prevalent

     in

     manufacturing

     and

     other

     industries

    .
    


    4

    .

     AI

     integration

     into

     entertainment

    :

    



```python
llm.shutdown()
```
