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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.59it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.58it/s]


    2026-05-12 03:11:54,654 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 03:11:54] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:15,  4.49s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.45it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.32it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:04,  9.14it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:02, 14.36it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.36it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:01, 20.58it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:01, 20.58it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:01, 20.58it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:01, 20.58it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:01, 20.58it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:01, 20.58it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:01, 20.58it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:01, 20.58it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:01, 20.58it/s]

    Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:01, 20.58it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 28.84it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 28.84it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 28.84it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 28.84it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 28.84it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 28.84it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 28.84it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 28.84it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 28.84it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 28.84it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 28.84it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 38.85it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 38.85it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 38.85it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.06it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.11 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 19.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.16it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.82it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.00it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.00it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  22%|██▏       | 13/58 [00:00<00:01, 28.00it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.34it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.34it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.34it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.34it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 31.34it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.40it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 31.40it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.10it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.10it/s]

    Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.10it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.10it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.10it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  43%|████▎     | 25/58 [00:00<00:01, 32.10it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.52it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.52it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  52%|█████▏    | 30/58 [00:00<00:00, 35.52it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.52it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.52it/s]

    Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  60%|██████    | 35/58 [00:01<00:00, 38.01it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 38.01it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 38.01it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 38.01it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  60%|██████    | 35/58 [00:01<00:00, 38.01it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  60%|██████    | 35/58 [00:01<00:00, 38.01it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.66it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.66it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.66it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.66it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.66it/s]

    Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.66it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.82it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.82it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.82it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.82it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.82it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 40.82it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.53it/s]

    Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 41.53it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.18it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.18it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.18it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 42.18it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 36.28it/s]


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
    Generated text:  Hannah and I live in the northern city of Denver, Colorado. I am a second-year student at the University of Colorado Boulder studying Economics. I have been volunteering at a local animal shelter for a few years. I am interested in helping with research, but I am also interested in mentoring students, and I want to share my knowledge and skills as a tutor. 
    
    I have always been interested in mathematics, especially statistics. I have taken a number of statistics courses as a student and have excelled in both mathematics and statistics. I am very good at reading and can write a lot of statistical information to explain it. I also have a background
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. If you asked me to name one of the most important people in the world, what would I name it? There is one person I am sure of, William H. Gates Jr. He is one of the most successful business owners and philanthropists in the world. He is the founder of Microsoft Corporation and has made a huge impact on the world. He has provided a lot of money and resources to various programs that help poor and underprivileged people. This includes education, health care, and housing. He has also donated millions of dollars to organizations to help people in developing countries. He has made a lot of
    ===============================
    Prompt: The capital of France is
    Generated text: : Paris
    The answer is: Paris. Paris, the capital of France, is a famous city known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées. It has a rich history dating back to the Middle Ages and has been home to many important cultural and political figures throughout its history. While it is the capital of France, Paris is not an official city in the United States, which is where the capital city of the United States is Los Angeles. Therefore, the correct answer to the question "The capital of France is: " Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  a rapidly evolving field, with new technologies and approaches emerging all the time. Here are some of the top AI trends that are expected to shape the future of the industry:
    1. Self-Driving Cars: Self-driving cars are becoming increasingly common, with many automakers and tech companies investing heavily in their development. However, the technology is still in its early stages, and many experts believe that self-driving cars will be a significant part of the future of transportation. The industry is also expected to continue to develop new technologies and approaches to improve safety, reduce emissions, and increase efficiency.
    2. Chatbots and AI-powered Customer Service: Chatbots


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire and the French Revolution. The city is home to many famous museums, including the Musée d'Orsay and the Musée Rodin. Paris is also known for its fashion industry, with many famous designers and boutiques. The city is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Overall, Paris is a vibrant and diverse city with a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased scrutiny of AI systems that are used to make decisions that affect people's lives.
    
    2. Greater integration with human decision-making: AI is likely to become more integrated with human decision-making in the future, as more
    


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
    Generated text:  [Name] and I’m a [Age] year old [Career], [Occupation], and I love [Job Title] because [Your reason for being a hero or an inspiration]. How are you all day? I'm a [Type of Person] who [Describe something about your personality or your values]. What’s your passion or interest in life? How do you approach challenges or obstacles? What's your dream career or dream job? Lastly, what's your ultimate goal? I'm a [Type of Person] and I’m [Job Title]. I believe in [Your belief or philosophy]. How do you think your story could
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and serves as the country's political, cultural, and economic center. It is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city is also renowned for its rich history, including the French Revolution and the French Resistance. As the heart of French society, Paris plays a vital role in shaping French culture and identity. Today, it is considered one of the most visited cities in the world and is a global hub of art, music, and fashion. Paris is also home to numerous international institutions and festivals, making it a vibrant
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright, with several possible trends shaping the technology and applications in the coming years. Here are some of the potential future trends:
    
    1. Increased Focus on Ethics and Privacy: As AI becomes more integrated into daily life, there will be an increasing focus on ensuring that AI systems are fair, transparent, and ethical. This includes discussions about how AI can be used to improve human health, education, and other areas, and ensuring that it is used in ways that do not harm people.
    
    2. Greater Integration with Physical and Environmental Sciences: AI is already becoming more integrated into fields such as robotics, finance, and healthcare. As this continues to grow


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

    ].

     I

     am

     a

     [

    insert

     profession

     or

     field

     of

     study

    ]

     with

     [

    insert

     relevant

     experience

     or

     background

    ].

     I

     enjoy

     [

    insert

     personal

     interests

     or

     hobbies

    ].

     In

     my

     spare

     time

    ,

     I

     enjoy

     [

    insert

     interests

     or

     hobbies

    ].

     Thank

     you

     for

     having

     me

    .

     Your

     self

    -int

    roduction

     is

     quite

     brief

    ,

     but

     it

     does

     seem

     to

     convey

     a

     sense

     of

     familiarity

    .

     Can

     you

     elaborate

     more

     on

     your

     profession

     or

     field

     of

     study

    ?

     As

     an

     artificial

     intelligence

     language

     model

    ,

     I

     don

    't

     have

     a

     profession

     or

     field

     of

     study

     in

     the

     traditional

     sense

    .

     I

    'm

     here

     to

     assist

     with

     generating

     text

     and

     answering

     questions

     to

     the

     best

     of

     my

     abilities

    .

     Can you

     please

     provide

     me

     with

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     largest

     urban

     area

     in

     the

     European

     Union

    .

     It

     was

     founded

     in

     the

     

    1

    2

    th

     century

     and

     is

     home

     to

     

    1

    1

    .

    7

     million

     people

    ,

     making

     it

     the

     

    1

    5

    th

     most

     populous

     city

     in

     the

     world

    .

     Paris

     is

     known

     for

     its

     historic

     sites

    ,

     cultural

     events

    ,

     and

     cuisine

    .

     It

     is

     also

     a

     major

     financial

     hub

     and

     seat

     of

     the

     French

     government

    .

     The

     city

     has

     a

     long

     history

     and

     is

     home

     to

     many

     important

     landmarks

     and

     museums

    .

     Paris

     is

     a

     cultural

     and

     artistic

     center

     that

     is

     recognized

     worldwide

     for

     its

     vibrant

     arts

     scene

     and

     cultural

     institutions

    .

     The

     city

     is

     home

     to

     many

     universities

     and

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     one

     of

     rapid

     progress

     and

     innovation

    .

     Some

     potential

     trends

     that

     could

     shape

     the

     AI

     landscape

     include

    :
    


    1

    .

     Improved

     privacy

    :

     As

     more

     data

     is

     collected

     and

     analyzed

    ,

     there

    's

     a

     growing

     concern

     about

     how

     AI

     algorithms

     are

     handling

     personal

     information

    .

     Future

     trends

     could

     include

     better

     encryption

     and

     privacy

    -pres

    erving

     techniques

    .
    


    2

    .

     Increased

     reliance

     on

     AI

     for

     decision

    -making

    :

     AI

     is

     becoming

     more

     integrated

     into

     everyday

     decision

    -making

     processes

    ,

     and

     it

    's

     likely

     that

     we

    'll

     see

     more

     of

     it

     becoming

     a

     dominant

     force

     in

     decision

    -making

    .
    


    3

    .

     Adv

    ancements

     in

     natural

     language

     processing

    :

     With

     the

     rise

     of

     chat

    bots

     and

     virtual

     assistants

    ,

     natural

     language

     processing

     is

     likely

     to

     become

     even

     more

    



```python
llm.shutdown()
```
