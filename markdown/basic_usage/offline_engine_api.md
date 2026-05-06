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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.81it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]


    2026-05-06 09:33:28,377 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-06 09:33:28] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:04,  4.29s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:03,  1.16s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:03,  1.16s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:03,  1.16s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.68it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:15,  3.25it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:07,  5.87it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:07,  5.87it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:07,  5.87it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:07,  5.87it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:07,  5.87it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:04<00:04,  8.92it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:04<00:04,  8.92it/s]

    Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:04<00:04,  8.92it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:04<00:04,  8.92it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:04<00:04,  8.92it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:05<00:04,  8.92it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:05<00:04,  8.92it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 14.66it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 14.66it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 14.66it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 14.66it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 14.66it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 14.66it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:02, 14.66it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:01, 20.72it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:01, 20.72it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:01, 20.72it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:01, 20.72it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:01, 20.72it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:01, 20.72it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:01, 20.72it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:01, 20.72it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:00, 28.63it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:00, 28.63it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:00, 28.63it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:00, 28.63it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:00, 28.63it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:00, 28.63it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:00, 28.63it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:00, 28.63it/s]

    Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:00, 28.63it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:05<00:00, 37.45it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:05<00:00, 37.45it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 43.77it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 43.77it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 43.77it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 43.77it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 43.77it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 43.77it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 43.77it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 43.77it/s] 

    Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 43.77it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.40it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=55.71 GB):   2%|▏         | 1/58 [00:00<00:09,  6.21it/s]Capturing num tokens (num_tokens=7680 avail_mem=55.68 GB):   2%|▏         | 1/58 [00:00<00:09,  6.21it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=55.68 GB):   3%|▎         | 2/58 [00:00<00:08,  6.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.67 GB):   3%|▎         | 2/58 [00:00<00:08,  6.36it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.67 GB):   5%|▌         | 3/58 [00:00<00:07,  7.28it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.67 GB):   5%|▌         | 3/58 [00:00<00:07,  7.28it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=55.67 GB):   7%|▋         | 4/58 [00:00<00:06,  7.76it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.67 GB):   7%|▋         | 4/58 [00:00<00:06,  7.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.66 GB):   7%|▋         | 4/58 [00:00<00:06,  7.76it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.66 GB):  10%|█         | 6/58 [00:00<00:05,  9.83it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.66 GB):  10%|█         | 6/58 [00:00<00:05,  9.83it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=55.65 GB):  10%|█         | 6/58 [00:00<00:05,  9.83it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.65 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.46it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.65 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.65 GB):  14%|█▍        | 8/58 [00:00<00:04, 11.46it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.65 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.07it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.64 GB):  17%|█▋        | 10/58 [00:00<00:03, 13.07it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=55.64 GB):  17%|█▋        | 10/58 [00:01<00:03, 13.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.64 GB):  21%|██        | 12/58 [00:01<00:03, 14.71it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.64 GB):  21%|██        | 12/58 [00:01<00:03, 14.71it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.64 GB):  21%|██        | 12/58 [00:01<00:03, 14.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.63 GB):  21%|██        | 12/58 [00:01<00:03, 14.71it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.63 GB):  26%|██▌       | 15/58 [00:01<00:02, 17.10it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.63 GB):  26%|██▌       | 15/58 [00:01<00:02, 17.10it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=55.62 GB):  26%|██▌       | 15/58 [00:01<00:02, 17.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.62 GB):  26%|██▌       | 15/58 [00:01<00:02, 17.10it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.62 GB):  31%|███       | 18/58 [00:01<00:02, 19.39it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.62 GB):  31%|███       | 18/58 [00:01<00:02, 19.39it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.62 GB):  31%|███       | 18/58 [00:01<00:02, 19.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.60 GB):  31%|███       | 18/58 [00:01<00:02, 19.39it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.60 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.59it/s]Capturing num tokens (num_tokens=960 avail_mem=55.61 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.59it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=55.61 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.59it/s]Capturing num tokens (num_tokens=832 avail_mem=55.60 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.59it/s]Capturing num tokens (num_tokens=768 avail_mem=55.60 GB):  36%|███▌      | 21/58 [00:01<00:01, 21.59it/s]Capturing num tokens (num_tokens=768 avail_mem=55.60 GB):  43%|████▎     | 25/58 [00:01<00:01, 24.52it/s]Capturing num tokens (num_tokens=704 avail_mem=55.60 GB):  43%|████▎     | 25/58 [00:01<00:01, 24.52it/s]Capturing num tokens (num_tokens=640 avail_mem=55.59 GB):  43%|████▎     | 25/58 [00:01<00:01, 24.52it/s]Capturing num tokens (num_tokens=576 avail_mem=55.59 GB):  43%|████▎     | 25/58 [00:01<00:01, 24.52it/s]Capturing num tokens (num_tokens=512 avail_mem=55.58 GB):  43%|████▎     | 25/58 [00:01<00:01, 24.52it/s]

    Capturing num tokens (num_tokens=512 avail_mem=55.58 GB):  50%|█████     | 29/58 [00:01<00:01, 26.71it/s]Capturing num tokens (num_tokens=480 avail_mem=55.59 GB):  50%|█████     | 29/58 [00:01<00:01, 26.71it/s]Capturing num tokens (num_tokens=448 avail_mem=55.59 GB):  50%|█████     | 29/58 [00:01<00:01, 26.71it/s]Capturing num tokens (num_tokens=416 avail_mem=55.59 GB):  50%|█████     | 29/58 [00:01<00:01, 26.71it/s]Capturing num tokens (num_tokens=384 avail_mem=55.59 GB):  50%|█████     | 29/58 [00:01<00:01, 26.71it/s]Capturing num tokens (num_tokens=384 avail_mem=55.59 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.58it/s]Capturing num tokens (num_tokens=352 avail_mem=55.58 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.58it/s]Capturing num tokens (num_tokens=320 avail_mem=55.58 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.58it/s]Capturing num tokens (num_tokens=288 avail_mem=55.58 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.58it/s]

    Capturing num tokens (num_tokens=256 avail_mem=55.57 GB):  57%|█████▋    | 33/58 [00:01<00:00, 28.58it/s]Capturing num tokens (num_tokens=256 avail_mem=55.57 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.93it/s]Capturing num tokens (num_tokens=240 avail_mem=55.57 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.93it/s]Capturing num tokens (num_tokens=224 avail_mem=55.57 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.93it/s]Capturing num tokens (num_tokens=208 avail_mem=55.56 GB):  64%|██████▍   | 37/58 [00:01<00:00, 29.93it/s]Capturing num tokens (num_tokens=192 avail_mem=55.56 GB):  64%|██████▍   | 37/58 [00:02<00:00, 29.93it/s]Capturing num tokens (num_tokens=192 avail_mem=55.56 GB):  71%|███████   | 41/58 [00:02<00:00, 30.33it/s]Capturing num tokens (num_tokens=176 avail_mem=55.56 GB):  71%|███████   | 41/58 [00:02<00:00, 30.33it/s]Capturing num tokens (num_tokens=160 avail_mem=55.56 GB):  71%|███████   | 41/58 [00:02<00:00, 30.33it/s]

    Capturing num tokens (num_tokens=144 avail_mem=55.55 GB):  71%|███████   | 41/58 [00:02<00:00, 30.33it/s]Capturing num tokens (num_tokens=128 avail_mem=55.55 GB):  71%|███████   | 41/58 [00:02<00:00, 30.33it/s]Capturing num tokens (num_tokens=128 avail_mem=55.55 GB):  78%|███████▊  | 45/58 [00:02<00:00, 24.71it/s]Capturing num tokens (num_tokens=112 avail_mem=55.55 GB):  78%|███████▊  | 45/58 [00:02<00:00, 24.71it/s]Capturing num tokens (num_tokens=96 avail_mem=55.54 GB):  78%|███████▊  | 45/58 [00:02<00:00, 24.71it/s] Capturing num tokens (num_tokens=80 avail_mem=55.54 GB):  78%|███████▊  | 45/58 [00:02<00:00, 24.71it/s]Capturing num tokens (num_tokens=64 avail_mem=55.54 GB):  78%|███████▊  | 45/58 [00:02<00:00, 24.71it/s]Capturing num tokens (num_tokens=64 avail_mem=55.54 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.72it/s]Capturing num tokens (num_tokens=48 avail_mem=55.53 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.72it/s]

    Capturing num tokens (num_tokens=32 avail_mem=55.53 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.72it/s]Capturing num tokens (num_tokens=28 avail_mem=55.52 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.72it/s]Capturing num tokens (num_tokens=24 avail_mem=55.52 GB):  84%|████████▍ | 49/58 [00:02<00:00, 26.72it/s]Capturing num tokens (num_tokens=24 avail_mem=55.52 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.54it/s]Capturing num tokens (num_tokens=20 avail_mem=55.52 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.54it/s]Capturing num tokens (num_tokens=16 avail_mem=55.52 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.54it/s]Capturing num tokens (num_tokens=12 avail_mem=55.51 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.54it/s]Capturing num tokens (num_tokens=8 avail_mem=55.51 GB):  91%|█████████▏| 53/58 [00:02<00:00, 28.54it/s] Capturing num tokens (num_tokens=8 avail_mem=55.51 GB):  98%|█████████▊| 57/58 [00:02<00:00, 30.61it/s]Capturing num tokens (num_tokens=4 avail_mem=55.51 GB):  98%|█████████▊| 57/58 [00:02<00:00, 30.61it/s]

    Capturing num tokens (num_tokens=4 avail_mem=55.51 GB): 100%|██████████| 58/58 [00:02<00:00, 21.78it/s]


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
    Generated text:  Cameron Riley. I am a 12 year old student who is living in his parents' home in Seattle, WA. I was born in November 1998 and graduated from Kingwood High School in October 2013. After attending college, I moved to Hawaii to pursue my passion for photography. While in Hawaii, I learned to photograph underwater and developed a love for marine life. I fell in love with photography after I captured the first photo of a whale.
    Recently, I have been working with my dad at his workplace in Seattle to develop a business called "Hawaii Porch". This is a photography business
    ===============================
    Prompt: The president of the United States is
    Generated text:  serving his 75th year and will be the 44th president to serve. How many presidents served between 25 and 40 years of age?
    To determine how many presidents served between 25 and 40 years of age, we need to identify the presidents who were born between 25 and 40 years of age. The president is currently serving his 75th year, which means he is 25 years younger than the president who is 44 years old.
    
    We need to count the number of presidents born between 25 and 40 years of age. Since
    ===============================
    Prompt: The capital of France is
    Generated text:  ________. A. Paris B. London C. Rome D. Moscow
    Answer:
    
    A
    
    The difference between the square of 7 and the square of 3 is ____
    A. 49
    B. 9
    C. 64
    D. 16
    Answer:
    
    A
    
    The Great Wall of China is the largest and most well-preserved ancient wall in the world. Building the Great Wall was completed during the ___ period.
    A. Spring and Autumn Warring States
    B. Western Zhou
    C. Spring and Autumn Warring States
    D. Western Han
    Answer:
    
    B
    
    According to the
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the people. In this modern era, as a digital AI development company, we are dedicated to helping businesses and enterprises innovate and transform their operations with the latest advancements in AI and machine learning. We offer a range of solutions, including machine learning algorithms, machine learning models, and AI tools that can be used to improve the efficiency, accuracy, and effectiveness of your business. Additionally, we offer a comprehensive range of services, including training, implementation, and support, to help you take advantage of the latest developments in AI and machine learning. So, whether you are a small business looking to improve its operations, or a large


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short description of your personality or skills that you're passionate about]. I enjoy [insert a short description of your hobbies or interests]. What do you like to do in your free time? I enjoy [insert a short description of your hobbies or interests]. What's your favorite hobby or activity? I love [insert a short description of your favorite hobby or activity]. What's your favorite book or movie? I love [insert a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, with a rich history and a diverse population. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is home to many famous museums, theaters, and restaurants, and is a major transportation hub for France and the European Union. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the coming years:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and improve the quality of care. As AI technology continues to improve, we can expect to see even more widespread adoption in healthcare, with more sophisticated algorithms and machine learning techniques being used to diagnose and treat diseases.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud
    


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
    Generated text: ... who are you? Hi there! I'm just an AI language model, and I don't have a name. How can I help you today? You might say, "I'm glad you're here, " or "Hello! I'm here to help! " Or you might say, "Hello, how can I assist you today? " or "Hello! What can I do for you?" The key is to be clear, concise, and friendly. Let me know if you have any specific topic or question you'd like to discuss, and I'll try my best to provide helpful and informative responses. Goodbye! And
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its historical significance, rich culture, and iconic landmarks like Notre-Dame Cathedral and the Eiffel Tower. Its vibrant nightlife, impressive museums, and architectural beauty make it a popular destination for tourists and locals alike. Paris offers a blend of old-world charm and modernity, making it a must-visit destination for anyone interested in French culture and history.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but here are some possible trends that could be expected in the coming years:
    
    1. Increased integration with other technologies: AI is already becoming more integrated with other technologies like the internet, the cloud, and IoT devices. It is expected that this integration will continue and expand, leading to a wider range of applications and services that can be powered by AI.
    
    2. Increased focus on ethical considerations: With the increasing number of AI-powered technologies in use, there is a growing focus on addressing ethical concerns and ensuring that AI is used in a responsible and fair manner.
    
    3. Advancements in natural language processing: Natural language processing (N


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

    Character

    's

     Name

    ],

     and

     I

     am

     a

     dedicated

     [

    Occup

    ation

    /

    Field

    ]

     with

     a

     passion

     for

     [

    Why

     is

     this

     field

     important

     to

     you

    ?

    ].

     My

     journey

     started

     when

     I

     [

    When

     did

     you

     first

     start

     learning

     this

     field

    ?

    ],

     and

     I

    've

     always

     been

     fascinated

     by

     [

    Why

     is

     this

     field

     interesting

     to

     you

    ?

    ].

     I

     love

     the

     challenge

     of

     [

    What

     is

     a

     key

     aspect

     of

     this

     field

     that

     makes

     it

     unique

    ?

    ].

     I

     am

     a

     [

    Your

     Character

    's

     Age

    ]

     year

     old

    ,

     and

     I

    'm

     constantly

     learning

     and

     growing

     as

     a

     person

    .

     I

     strive

     to

     be

     [

    What

     character

     trait

     do

     you

     have

     that

     drives

     you

     to

     succeed

    ?

    ].

     I

     am

     always

     eager

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     located

     on

     the

     north

     bank

     of

     the

     Se

    ine

     River

    .

     It

     is

     known

     as

     "

    la

     ville

     environ

    n

    ante

    "

     and

     is

     home

     to

     numerous

     landmarks

    ,

     including

     the

     E

    iff

    el

     Tower

    ,

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     known

     for

     its

     vibrant

     nightlife

     and

     jazz

     scene

    ,

     and

     is

     a

     UNESCO

     World

     Heritage

     Site

    .

     It

     is

     a

     bustling

     city

     with

     a

     rich

     history

     and

     a

     diverse

     cultural

     landscape

    .

     It

     is

     a

     city

     of

     contrasts

    ,

     with

     traditional

     French

     architecture

     and

     modern

     amenities

     in

     between

    .

     Overall

    ,

     Paris

     is

     an

     iconic

     city

     that

     has

     played

     a

     significant

     role

     in

     the

     history

     and

     development

     of

     France

    .

     Paris

     is

     often

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     highly

     dependent

     on

     a

     combination

     of

     technological

     advances

    ,

     societal

     changes

    ,

     and

     economic

     shifts

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     AI

     Integration

    :

     As

     AI

     becomes

     more

     capable

     and

     widespread

    ,

     it

     is

     likely

     to

     become

     even

     more

     integrated

     into

     our

     daily

     lives

    .

     This

     could

     mean

     more

     use

     cases

     for

     AI

     in

     healthcare

    ,

     finance

    ,

     education

    ,

     and

     more

    .
    


    2

    .

     Personal

    ized

     AI

    :

     With

     more

     data

     being

     collected

     and

     analyzed

    ,

     AI

     is

     likely

     to

     become

     even

     more

     personalized

    .

     This

     could

     lead

     to

     more

     intelligent

     algorithms

     that

     can

     learn

     and

     adapt

     to

     individual

     user

     needs

    .
    


    3

    .

     Autonomous

     AI

    :

     Self

    -driving

     cars

    



```python
llm.shutdown()
```
