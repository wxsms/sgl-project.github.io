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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.24it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.23it/s]


    2026-04-11 09:25:41,321 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 09:25:41] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:36,  2.74s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:02<00:41,  1.33it/s]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:02<00:41,  1.33it/s]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:02<00:41,  1.33it/s]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:02<00:41,  1.33it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:02<00:16,  3.21it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:02<00:16,  3.21it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:02<00:16,  3.21it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:03<00:16,  3.21it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:03<00:16,  3.21it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:03<00:16,  3.21it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:03<00:06,  7.15it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:03<00:06,  7.15it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:03<00:06,  7.15it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:03<00:06,  7.15it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:03<00:06,  7.15it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:03<00:06,  7.15it/s]

    Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:06,  7.15it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:03<00:03, 12.85it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:03<00:03, 12.85it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:01, 19.09it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:01, 19.09it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:01, 19.09it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:01, 19.09it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:01, 19.09it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:03<00:01, 19.09it/s]

    Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:03<00:01, 19.09it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 25.28it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 25.28it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 25.28it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 25.28it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 25.28it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 25.28it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 25.28it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:03<00:01, 25.28it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:03<00:00, 31.96it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:03<00:00, 31.96it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:03<00:00, 31.96it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:03<00:00, 31.96it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:03<00:00, 31.96it/s]

    Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:03<00:00, 31.96it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:03<00:00, 31.96it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:03<00:00, 31.96it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:03<00:00, 39.39it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:03<00:00, 39.39it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:03<00:00, 39.39it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:03<00:00, 39.39it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:03<00:00, 39.39it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:03<00:00, 39.39it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:03<00:00, 39.39it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:03<00:00, 39.39it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 45.35it/s]

    Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 45.35it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 45.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.72 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=74.72 GB):   2%|▏         | 1/58 [00:00<00:28,  1.97it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.69 GB):   2%|▏         | 1/58 [00:00<00:28,  1.97it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.69 GB):   3%|▎         | 2/58 [00:00<00:17,  3.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.39 GB):   3%|▎         | 2/58 [00:00<00:17,  3.24it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=74.39 GB):   5%|▌         | 3/58 [00:00<00:11,  4.59it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.39 GB):   5%|▌         | 3/58 [00:00<00:11,  4.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.39 GB):   5%|▌         | 3/58 [00:00<00:11,  4.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:00<00:07,  6.73it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.38 GB):   9%|▊         | 5/58 [00:00<00:07,  6.73it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=74.39 GB):   9%|▊         | 5/58 [00:01<00:07,  6.73it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.39 GB):  12%|█▏        | 7/58 [00:01<00:05,  8.82it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.38 GB):  12%|█▏        | 7/58 [00:01<00:05,  8.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.38 GB):  12%|█▏        | 7/58 [00:01<00:05,  8.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.95it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.95it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.37 GB):  16%|█▌        | 9/58 [00:01<00:04, 10.95it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.94it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.37 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.94it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.36 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.36 GB):  19%|█▉        | 11/58 [00:01<00:03, 12.94it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.36 GB):  24%|██▍       | 14/58 [00:01<00:02, 15.22it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.35 GB):  24%|██▍       | 14/58 [00:01<00:02, 15.22it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=74.35 GB):  24%|██▍       | 14/58 [00:01<00:02, 15.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.35 GB):  24%|██▍       | 14/58 [00:01<00:02, 15.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.35 GB):  29%|██▉       | 17/58 [00:01<00:02, 17.49it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:01<00:02, 17.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:01<00:02, 17.49it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:01<00:02, 17.49it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.32 GB):  29%|██▉       | 17/58 [00:01<00:02, 17.49it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.32 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.97it/s]Capturing num tokens (num_tokens=960 avail_mem=74.33 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.97it/s] Capturing num tokens (num_tokens=896 avail_mem=74.33 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.97it/s]Capturing num tokens (num_tokens=832 avail_mem=74.33 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.97it/s]Capturing num tokens (num_tokens=832 avail_mem=74.33 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=768 avail_mem=74.32 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=704 avail_mem=74.32 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=640 avail_mem=74.32 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=576 avail_mem=74.32 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.86it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.30 GB):  41%|████▏     | 24/58 [00:01<00:01, 23.86it/s]Capturing num tokens (num_tokens=512 avail_mem=74.30 GB):  50%|█████     | 29/58 [00:01<00:00, 29.23it/s]Capturing num tokens (num_tokens=480 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:01<00:00, 29.23it/s]Capturing num tokens (num_tokens=448 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:01<00:00, 29.23it/s]Capturing num tokens (num_tokens=416 avail_mem=74.32 GB):  50%|█████     | 29/58 [00:01<00:00, 29.23it/s]Capturing num tokens (num_tokens=384 avail_mem=74.31 GB):  50%|█████     | 29/58 [00:02<00:00, 29.23it/s]Capturing num tokens (num_tokens=352 avail_mem=74.31 GB):  50%|█████     | 29/58 [00:02<00:00, 29.23it/s]Capturing num tokens (num_tokens=320 avail_mem=74.30 GB):  50%|█████     | 29/58 [00:02<00:00, 29.23it/s]Capturing num tokens (num_tokens=320 avail_mem=74.30 GB):  60%|██████    | 35/58 [00:02<00:00, 35.89it/s]Capturing num tokens (num_tokens=288 avail_mem=74.30 GB):  60%|██████    | 35/58 [00:02<00:00, 35.89it/s]Capturing num tokens (num_tokens=256 avail_mem=74.30 GB):  60%|██████    | 35/58 [00:02<00:00, 35.89it/s]Capturing num tokens (num_tokens=240 avail_mem=74.30 GB):  60%|██████    | 35/58 [00:02<00:00, 35.89it/s]

    Capturing num tokens (num_tokens=224 avail_mem=74.29 GB):  60%|██████    | 35/58 [00:02<00:00, 35.89it/s]Capturing num tokens (num_tokens=224 avail_mem=74.29 GB):  67%|██████▋   | 39/58 [00:02<00:00, 35.94it/s]Capturing num tokens (num_tokens=208 avail_mem=74.29 GB):  67%|██████▋   | 39/58 [00:02<00:00, 35.94it/s]Capturing num tokens (num_tokens=192 avail_mem=74.29 GB):  67%|██████▋   | 39/58 [00:02<00:00, 35.94it/s]Capturing num tokens (num_tokens=176 avail_mem=74.26 GB):  67%|██████▋   | 39/58 [00:02<00:00, 35.94it/s]Capturing num tokens (num_tokens=160 avail_mem=74.00 GB):  67%|██████▋   | 39/58 [00:02<00:00, 35.94it/s]Capturing num tokens (num_tokens=160 avail_mem=74.00 GB):  74%|███████▍  | 43/58 [00:02<00:00, 32.38it/s]Capturing num tokens (num_tokens=144 avail_mem=74.24 GB):  74%|███████▍  | 43/58 [00:02<00:00, 32.38it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.24 GB):  74%|███████▍  | 43/58 [00:02<00:00, 32.38it/s]Capturing num tokens (num_tokens=112 avail_mem=74.24 GB):  74%|███████▍  | 43/58 [00:02<00:00, 32.38it/s]Capturing num tokens (num_tokens=96 avail_mem=74.24 GB):  74%|███████▍  | 43/58 [00:02<00:00, 32.38it/s] Capturing num tokens (num_tokens=96 avail_mem=74.24 GB):  81%|████████  | 47/58 [00:02<00:00, 28.85it/s]Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  81%|████████  | 47/58 [00:02<00:00, 28.85it/s]Capturing num tokens (num_tokens=64 avail_mem=74.10 GB):  81%|████████  | 47/58 [00:02<00:00, 28.85it/s]

    Capturing num tokens (num_tokens=48 avail_mem=74.22 GB):  81%|████████  | 47/58 [00:02<00:00, 28.85it/s]Capturing num tokens (num_tokens=32 avail_mem=74.22 GB):  81%|████████  | 47/58 [00:02<00:00, 28.85it/s]Capturing num tokens (num_tokens=32 avail_mem=74.22 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.46it/s]Capturing num tokens (num_tokens=28 avail_mem=74.20 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.46it/s]Capturing num tokens (num_tokens=24 avail_mem=74.20 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.46it/s]Capturing num tokens (num_tokens=20 avail_mem=74.19 GB):  88%|████████▊ | 51/58 [00:02<00:00, 28.46it/s]Capturing num tokens (num_tokens=20 avail_mem=74.19 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.69it/s]Capturing num tokens (num_tokens=16 avail_mem=74.19 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.69it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.18 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.69it/s]Capturing num tokens (num_tokens=8 avail_mem=74.17 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.69it/s] Capturing num tokens (num_tokens=4 avail_mem=74.17 GB):  93%|█████████▎| 54/58 [00:02<00:00, 28.69it/s]Capturing num tokens (num_tokens=4 avail_mem=74.17 GB): 100%|██████████| 58/58 [00:02<00:00, 29.87it/s]Capturing num tokens (num_tokens=4 avail_mem=74.17 GB): 100%|██████████| 58/58 [00:02<00:00, 20.14it/s]


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
    Generated text:  Alex. I am an 11-year-old boy from West Los Angeles. My name is "Alex". It's a short name that is easy to remember and pronounce. Is there anything else you would like to know about me? Or do you have any other questions? Goodbye! (End of message) Can you provide more information about the short name "Alex"? Sure, I'd be happy to provide more information about the short name "Alex". Is there anything specific you would like to know about Alex? Or do you have any other questions? Goodbye! (End of message) Can you tell me about the short name
    ===============================
    Prompt: The president of the United States is
    Generated text:  elected in an election that determines the winner of that state's congressional election. The president then nominates a running mate to the position of the U.S. president, which is ultimately decided by the Senate. If the Senate is divided into three equal parts, and two senators leave the Senate and one senator remains, who becomes the president of the United States? To determine the president of the United States under the given conditions, let's break down the process step by step:
    
    1. **Electing the President in the State's Congressional Election:**
       - The president is elected in the state's congressional election.
       - This election determines the
    ===============================
    Prompt: The capital of France is
    Generated text:  located in the heart of the Alpine region. The capital of Switzerland is located in the Alps. 
    
    Question: Where is the capital of Switzerland located?
    A) Mountainous region
    B) Alpine region
    C) Coastal region
    D) Mountainous region of Italy
    E) Alpine region of Italy The capital of Switzerland is located in the Alps, not in the Mediterranean or in a coastal area. Therefore, the correct answer is:
    
    B) Alpine region
    
    The capital of Switzerland is located in the northern part of the Alps, which is known as Bern. Bern is a city in the Swiss canton of Jura. It is known
    ===============================
    Prompt: The future of AI is
    Generated text:  4.0, with the increasing prevalence of big data and the advent of the Internet of Things (IoT). With the advancement of IoT, we are facing more and more opportunities for AI in various fields. In the following, we will cover the four key areas of AI applications in the future.
    1. Healthcare
    AI has a huge impact on healthcare in the future, and the roles of AI in this field are expected to continue to expand.
    - AI can analyze medical imaging, such as CT and MRI, to help doctors determine the cause of diseases and improve the diagnostic accuracy of diagnoses.
    - AI can predict the risk of diseases


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest in the company or industry]. I enjoy [reason for interest in the company or industry] and I'm always looking for ways to [reason for interest in the company or industry]. I'm a [reason for interest in the company or industry] and I'm always looking for ways to [reason for interest in the company or industry]. I'm a [reason for interest in the company or industry] and I'm always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history dating back to the Middle Ages. It is located in the south of France and is the largest city in the country. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its diverse cultural scene and food scene. The city is also home to many famous museums, including the Musée d'Orsay and the Musée d'Orsay. Paris is a popular tourist destination and is known for its fashion, art, and cuisine. It is also home to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI systems, there is likely to be a greater emphasis on ethical considerations and the development of AI that is designed to be fair, transparent, and accountable.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, with machines being able to learn and adapt based on the feedback and experiences of humans. This could lead to more effective and efficient decision
    


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
    Generated text:  [insert character name], and I'm an [insert character's age]. I'm [insert character's occupation], and I've always been fascinated by [insert a brief quote about my hobby or interest]. I've always been driven by [insert a short summary of why I like or love what I do]. I've always been excited to [insert a short description of why I'm excited about my job or hobby]. I enjoy [insert a short summary of my hobbies or interests], and I love spending time [insert a short description of how I like to spend my time]. I'm [insert a short description of my personality]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That's correct! Paris is the capital city of France. It's a sprawling city with a rich history and a vibrant culture. Here are some key facts about Paris:
    
    1. Population: Over 2 million people live in Paris.
    
    2. World Famous: Paris is home to landmarks such as the Eiffel Tower, Louvre Museum, and Notre Dame Cathedral.
    
    3. Food: It's famous for its cuisine, including French dishes like croissants, beignets, and pastries.
    
    4. Architecture: Many iconic buildings in Paris, including the Louvre, Notre Dame, and Eiffel Tower, are
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be marked by a rapid expansion in its capabilities and applications, as well as a growing awareness of the ethical and social implications of AI technologies. Here are some possible future trends in AI:
    
    1. Increased integration with human experience: As AI becomes more integrated into our daily lives, we may see an increase in its ability to interact with humans in new and innovative ways. For example, AI-powered virtual assistants and chatbots may become more effective at understanding and responding to human speech, and may even be able to provide personalized recommendations for tasks and activities.
    
    2. Enhanced human-computer interaction: AI technologies may also be able to enhance human


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

    Your

     Profession

     or

     Type

     of

     Character

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

     have

     always

     been

     passionate

     about

     [

    Your

     Interest

     or

     Focus

    ].

     I

     believe

     that

     [

    Your

     Hobby

     or

     Passion

    ]

     is

     the

     best

     way

     to

     make

     me

     a

     complete

     person

    .

     I

     am

     [

    Your

     Character

    istic

     or

     Unique

     Traits

    ].

     I

     love

     [

    Your

     Hobby

     or

     Passion

    ]

     because

     [

    Reason

     for

     Your

     Passion

    ].

     I

     am

     always

     up

     for

     [

    Your

     Job

     Responsibilities

     or

     Hard

     Work

    /

    Th

    irst

     for

     Knowledge

    ].

     I

     am

     confident

     in

     my

     [

    Your

     Strength

     or

     Skill

    ].

     I

     am

     always

     looking

     for

     [

    Your

     Goal

     or

     Mot

    ivation

    /

    Challenge

    ].

     Thank

     you

     for

     having

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     the

     third

     largest

     in

     the

     world

    .

     Paris

     is

     known

     for

     its

     cultural

     institutions

    ,

     renowned

     landmarks

    ,

     and

     historic

     sites

    ,

     including

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

     Ch

    amps

    -

    É

    lys

    ées

    .

     Paris

     also

     boasts

     a

     vibrant

     arts

     and

     culture

     scene

    ,

     attracting

     many

     artists

     and

     musicians

     from

     around

     the

     world

    .

     The

     city

     is

     also

     known

     for

     its

     distinctive

     cuisine

    ,

     including

     cro

    iss

    ants

     and

     bag

    uet

    tes

    ,

     as

     well

     as

     its

     iconic

     past

    ries

     such

     as

     cro

    iss

    ants

     and

     cro

    que

    -m

    ons

    ieur

    .

     Paris

     is

     a

     major

     hub

     for

     business

    ,

     finance

    ,

     and

     tourism

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     exciting

     and

     diverse

    .

     Here

     are

     some

     possible

     trends

    :
    


    1

    .

     Increased

     sophistication

    :

     AI

     will

     continue

     to

     improve

     and

     become

     even

     more

     sophisticated

    ,

     with

     the

     ability

     to

     learn

     and

     adapt

     to

     new

     situations

    .

     This

     will

     allow

     AI

     to

     become

     more

     efficient

     and

     effective

     at

     performing

     tasks

    .
    


    2

    .

     Autonomous

     vehicles

    :

     AI

     has

     already

     been

     used

     in

     self

    -driving

     cars

    ,

     and

     it

     is

     expected

     that

     this

     technology

     will

     become

     more

     advanced

     in

     the

     future

    .

     Autonomous

     vehicles

     will

     likely

     have

     the

     ability

     to

     understand

     and

     respond

     to

     the

     driver

    's

     commands

    ,

     making

     them

     safer

     and

     more

     reliable

    .
    


    3

    .

     Personal

    ized

     healthcare

    :

     AI

     will

     be

     used

     to

     analyze

     patient

     data

     and

     provide

     personalized

     treatment

     plans

    .

     This

     will

    



```python
llm.shutdown()
```
