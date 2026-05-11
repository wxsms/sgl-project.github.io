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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.28it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.27it/s]


    2026-05-11 21:24:02,605 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-11 21:24:02] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:02,  4.26s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:45,  1.20it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:14,  3.35it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:14,  3.35it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:14,  3.35it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:14,  3.35it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:14,  3.35it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:14,  3.35it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:14,  3.35it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:14,  3.35it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:04<00:14,  3.35it/s]Compiling num tokens (num_tokens=1792):  16%|█▌        | 9/58 [00:04<00:14,  3.35it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:04<00:04,  8.47it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:04<00:04,  8.47it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:04<00:04,  8.47it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:04<00:04,  8.47it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:04<00:04,  8.47it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:04<00:04,  8.47it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:04<00:04,  8.47it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:04<00:02, 12.46it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:04<00:02, 12.46it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:04<00:02, 12.46it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:04<00:02, 12.46it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:04<00:02, 12.46it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:04<00:02, 12.46it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:04<00:02, 12.46it/s]Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:04<00:02, 12.46it/s]Compiling num tokens (num_tokens=416):  41%|████▏     | 24/58 [00:04<00:02, 12.46it/s]Compiling num tokens (num_tokens=384):  41%|████▏     | 24/58 [00:04<00:02, 12.46it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:04<00:01, 20.17it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:04<00:01, 20.17it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:04<00:01, 20.17it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:04<00:01, 20.17it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:04<00:01, 20.17it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:04<00:01, 20.17it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:04<00:01, 20.17it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:04<00:01, 20.17it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:04<00:01, 20.17it/s]Compiling num tokens (num_tokens=176):  57%|█████▋    | 33/58 [00:04<00:01, 20.17it/s]

    Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 28.51it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 28.51it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 28.51it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 28.51it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 28.51it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 28.51it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 28.51it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 28.51it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 28.51it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 35.26it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 35.26it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 35.26it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 35.26it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 35.26it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 35.26it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 35.26it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 35.26it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 35.26it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.67 GB):   3%|▎         | 2/58 [00:00<00:04, 13.76it/s]Capturing num tokens (num_tokens=7168 avail_mem=73.59 GB):   3%|▎         | 2/58 [00:00<00:04, 13.76it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=73.66 GB):   3%|▎         | 2/58 [00:00<00:04, 13.76it/s]Capturing num tokens (num_tokens=6656 avail_mem=73.66 GB):   7%|▋         | 4/58 [00:00<00:03, 15.23it/s]Capturing num tokens (num_tokens=6144 avail_mem=73.65 GB):   7%|▋         | 4/58 [00:00<00:03, 15.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.64 GB):   7%|▋         | 4/58 [00:00<00:03, 15.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=73.64 GB):  10%|█         | 6/58 [00:00<00:03, 16.93it/s]Capturing num tokens (num_tokens=5120 avail_mem=73.63 GB):  10%|█         | 6/58 [00:00<00:03, 16.93it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=73.59 GB):  10%|█         | 6/58 [00:00<00:03, 16.93it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.60 GB):  10%|█         | 6/58 [00:00<00:03, 16.93it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.60 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.45it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.60 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.45it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.59 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.45it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.58 GB):  16%|█▌        | 9/58 [00:00<00:02, 19.45it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.58 GB):  21%|██        | 12/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.58 GB):  21%|██        | 12/58 [00:00<00:02, 22.68it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=73.57 GB):  21%|██        | 12/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.56 GB):  21%|██        | 12/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.57 GB):  21%|██        | 12/58 [00:00<00:02, 22.68it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.57 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.35it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.58 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.35it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.55 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.35it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.57 GB):  28%|██▊       | 16/58 [00:00<00:01, 26.35it/s]Capturing num tokens (num_tokens=1536 avail_mem=73.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.41it/s]Capturing num tokens (num_tokens=1280 avail_mem=73.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.41it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=73.53 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.41it/s]Capturing num tokens (num_tokens=960 avail_mem=73.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.41it/s] Capturing num tokens (num_tokens=896 avail_mem=73.54 GB):  33%|███▎      | 19/58 [00:00<00:01, 27.41it/s]Capturing num tokens (num_tokens=896 avail_mem=73.54 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.77it/s]Capturing num tokens (num_tokens=832 avail_mem=73.53 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.77it/s]Capturing num tokens (num_tokens=768 avail_mem=73.53 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.77it/s]Capturing num tokens (num_tokens=704 avail_mem=73.52 GB):  40%|███▉      | 23/58 [00:00<00:01, 30.77it/s]Capturing num tokens (num_tokens=640 avail_mem=73.52 GB):  40%|███▉      | 23/58 [00:01<00:01, 30.77it/s]Capturing num tokens (num_tokens=640 avail_mem=73.52 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.48it/s]Capturing num tokens (num_tokens=576 avail_mem=73.51 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.48it/s]

    Capturing num tokens (num_tokens=512 avail_mem=73.49 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.48it/s]Capturing num tokens (num_tokens=480 avail_mem=73.51 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.48it/s]Capturing num tokens (num_tokens=448 avail_mem=73.50 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.48it/s]Capturing num tokens (num_tokens=416 avail_mem=73.50 GB):  47%|████▋     | 27/58 [00:01<00:00, 33.48it/s]Capturing num tokens (num_tokens=416 avail_mem=73.50 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.62it/s]Capturing num tokens (num_tokens=384 avail_mem=73.49 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.62it/s]Capturing num tokens (num_tokens=352 avail_mem=73.48 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.62it/s]Capturing num tokens (num_tokens=320 avail_mem=73.47 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.62it/s]Capturing num tokens (num_tokens=288 avail_mem=73.47 GB):  55%|█████▌    | 32/58 [00:01<00:00, 35.62it/s]

    Capturing num tokens (num_tokens=288 avail_mem=73.47 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=256 avail_mem=73.46 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=240 avail_mem=73.45 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=224 avail_mem=73.45 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=208 avail_mem=73.44 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=192 avail_mem=73.44 GB):  62%|██████▏   | 36/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=192 avail_mem=73.44 GB):  71%|███████   | 41/58 [00:01<00:00, 37.49it/s]Capturing num tokens (num_tokens=176 avail_mem=73.43 GB):  71%|███████   | 41/58 [00:01<00:00, 37.49it/s]Capturing num tokens (num_tokens=160 avail_mem=73.42 GB):  71%|███████   | 41/58 [00:01<00:00, 37.49it/s]

    Capturing num tokens (num_tokens=144 avail_mem=73.42 GB):  71%|███████   | 41/58 [00:01<00:00, 37.49it/s]Capturing num tokens (num_tokens=128 avail_mem=73.41 GB):  71%|███████   | 41/58 [00:01<00:00, 37.49it/s]Capturing num tokens (num_tokens=128 avail_mem=73.41 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=112 avail_mem=73.41 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=96 avail_mem=73.41 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.83it/s] Capturing num tokens (num_tokens=80 avail_mem=73.40 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=64 avail_mem=73.40 GB):  78%|███████▊  | 45/58 [00:01<00:00, 34.83it/s]Capturing num tokens (num_tokens=64 avail_mem=73.40 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.75it/s]Capturing num tokens (num_tokens=48 avail_mem=73.40 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.75it/s]Capturing num tokens (num_tokens=32 avail_mem=73.39 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.75it/s]

    Capturing num tokens (num_tokens=28 avail_mem=73.39 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.75it/s]Capturing num tokens (num_tokens=24 avail_mem=73.39 GB):  84%|████████▍ | 49/58 [00:01<00:00, 35.75it/s]Capturing num tokens (num_tokens=24 avail_mem=73.39 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.94it/s]Capturing num tokens (num_tokens=20 avail_mem=73.38 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.94it/s]Capturing num tokens (num_tokens=16 avail_mem=73.38 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.94it/s]Capturing num tokens (num_tokens=12 avail_mem=73.38 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.94it/s]Capturing num tokens (num_tokens=8 avail_mem=73.37 GB):  91%|█████████▏| 53/58 [00:01<00:00, 35.94it/s] Capturing num tokens (num_tokens=8 avail_mem=73.37 GB):  98%|█████████▊| 57/58 [00:01<00:00, 34.77it/s]Capturing num tokens (num_tokens=4 avail_mem=73.37 GB):  98%|█████████▊| 57/58 [00:01<00:00, 34.77it/s]Capturing num tokens (num_tokens=4 avail_mem=73.37 GB): 100%|██████████| 58/58 [00:01<00:00, 30.88it/s]


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
    Generated text:  Max. I am a computer science student who has been working with JavaScript for a few months now, and I'm interested in learning about statistical analysis and machine learning. Could you please give me some advice on how to optimize my project for better performance? Sure, I'd be happy to help! To start, you should start by looking at your code and identifying any potential bottlenecks. Is there a specific part of the code that you think is causing a performance issue? Once you've identified the problem, you can work on optimizing it by breaking it down into smaller, more manageable parts. This will make it easier to find the root
    ===============================
    Prompt: The president of the United States is
    Generated text:  expected to have the most power in the world. In fact, he is expected to be the most powerful man in the world. The world’s most powerful man is the president of the United States. There have been many presidents in the past. No one has been president longer than the President of the United States. The president is elected by a large group of people. The president is the head of the government of the United States. He is the leader of the United States. He is the president of the United States. The president of the United States is the most powerful man in the world. But the president of the United States has
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. A) Rome B) Paris C) London D) Athens
    Answer:
    
    B) Paris. Paris is the capital of France, and it is also the largest city in France. It is located on the western coast of the French mainland and is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous attractions such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, as well as the romantic Eiffel Tower, a structure that has been in continuous use since 1889. In addition, the city is known for its French cuisine, fashion,
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the individuals who will develop it and shape its course. This is the philosophy of AI-driven innovation, which has been driving the evolution of the field of artificial intelligence (AI) to new heights in recent years. AI has been a powerful tool that has been widely used to automate processes and improve efficiency, making them more efficient and effective. The use of AI has also led to the creation of new industries and the development of new technologies, which have transformed the way we live and work.
    One of the key areas where AI is having a significant impact is in healthcare. The field of AI is being used to develop new diagnostic


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender] with [Number] years of experience in [Field of Work]. I'm a [Number] year old, [Gender]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, and is the largest city in the country and the seat of the French government. It is located on the Seine River and is home to many of France's most famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its rich cultural heritage, including the city's famous museums, theaters, and art galleries. The city is known for its fashion industry, with many famous fashion designers and boutiques located in the city. Paris is a vibrant and dynamic city with a rich history and a diverse population. It is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be an increased focus on ethical considerations. This includes issues such as bias, transparency, and accountability. AI developers will need to ensure that their systems are designed to be fair and unbiased, and that they are transparent and explainable.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This will allow
    


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
    Generated text:  [Your Name] and I'm a [career] with [number] years of experience. [Your Name] is currently employed as [job title] and has a [number] of years of experience in this position. I am always looking to improve and continue learning as I strive to make a positive impact in the field.
    In addition to my career, I am also passionate about [your hobby, interest, or passion], and I strive to incorporate this into my work through [details about how you apply this to your career].
    I am confident that my skills, experience, and passions make me a strong candidate for any role I may
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "la Parísne" and "Paris".
    
    The French capital is situated in the southwestern part of France, at the mouth of the Seine River, just north of the Ile de France. Its name is a combination of the French words for "city" and "air", in reference to its strategic location and importance as a gateway to Europe.
    
    The city is home to one of the world's most popular art museums, the Louvre, and is the birthplace of many notable figures such as Leonardo da Vinci, Oscar Wilde, and Camille Pissarro. The city is also known for its cultural
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, and it's likely to continue expanding and advancing in many different ways. Some possible future trends in AI include:
    
    1. Increased automation: As AI becomes more sophisticated, it's likely to automate more tasks, freeing up human workers and allowing for greater automation in other areas.
    
    2. Enhanced human capabilities: AI may become even more capable of performing tasks that were previously done by humans, such as natural language processing, image recognition, and decision-making.
    
    3. Increased integration with human emotions: AI may become more integrated with human emotions and consciousness, allowing for more empathetic and emotional AI that can understand and respond to human emotions.
    
    


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

     [

    Age

    ] years

     old

    .

     I

    've

     always

     been

     a

     hard

    working

     and

     ambitious

     person

    ,

     always

     striving

     to

     achieve

     my

     goals

    .

     I

    'm

     always

     on

     the

     go

     and

     don

    't

     take

     things

     lightly

    .

     I

     enjoy

     solving

     problems

     and

     finding

     innovative

     solutions

    ,

     which

     I

     believe

     is

     key

     to

     success

    .

     I

     believe

     in

     pursuing

     my

     passions

    ,

     whether

     that

    's

     music

    ,

     sports

    ,

     or

     writing

    ,

     and

     I

    'm

     always

     up

     for

     the

     challenge

    .

     I

    'm

     excited

     to

     have

     the

     opportunity

     to

     meet

     you

     and

     learn

     more

     about

     you

    .

     How

     about

     you

    ?

     What

     brings

     you

     here

    ?

     ...

    Continue

     writing

    .


    As

     you

     embark

     on

     this

     journey

    ,

     I

    'm

     eager

     to

     get

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Grande

     E

    to

    ile

    "

     and

     the

     "

    City

     of

     Light

    ".

     It

     is

     the

     largest

     city

     in

     Europe

     and

     one

     of

     the

     largest

     in

     the

     world

    ,

     with

     over

     

    1

    .

     

    5

     million

     residents

     as

     of

     

    2

    0

    1

    7

    . The

     city

     is

     located

     in

     the

     south

    -west

    ern

     part

     of

     France

    ,

     on

     the

     Î

    le

     de

     France

    ,

     and

     is

     a

     UNESCO

     World

     Heritage

     site

    .

     It

     is

     known

     for

     its

     towering

     architecture

    ,

     beautiful

     gardens

    ,

     and

     vibrant

     culture

    ,

     making

     it

     a

     popular

     tourist

     destination

    .

     Paris

     is

     also

     home

     to

     many

     world

    -ren

    owned

     museums

    ,

     including

     the

     Lou

    vre

    ,

     the

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Mus

    ée

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     characterized

     by

     significant

     advancements

     in

     areas

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     robotics

    ,

     and

     autonomous

     systems

    .

     Here

     are

     some

     potential

     future

     trends

     that

     may

     shape

     the

     AI

     landscape

    :
    


    1

    .

     Increased

     focus

     on

     ethics

     and

     fairness

    :

     As

     AI

     systems

     are

     becoming

     more

     complex

     and

     interconnected

    ,

     it

     is

     likely

     that

     ethical

     considerations

     will

     become

     increasingly

     important

    .

     We

     may

     see

     increased

     focus

     on

     ensuring

     that

     AI

     systems

     are

     designed

     and

     implemented

     in

     ways

     that

     benefit everyone

    ,

     rather

     than just

     the

     people

     who

     built

     them

    .
    


    2

    .

     Integration

     of

     AI

     with

     human

     decision

    -making

    :

     As

     AI

     becomes

     more

     integrated

     into

     decision

    -making

     processes

    ,

     we

     may

     see

     a

     shift

     towards

     more

     human

    -like

     AI

     that

     can

    



```python
llm.shutdown()
```
