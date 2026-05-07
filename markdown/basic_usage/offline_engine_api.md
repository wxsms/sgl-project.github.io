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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.83it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.83it/s]


    2026-05-07 13:43:06,707 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 13:43:06] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]

    Compiling num tokens (num_tokens=480):  34%|███▍      | 20/58 [00:04<00:04,  8.75it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.55it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 23.72it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 23.72it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 23.72it/s]

    Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 23.72it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 32.83it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 32.83it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 32.83it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 32.83it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 32.83it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 32.83it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 32.83it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 32.83it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 32.83it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.38it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.15 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.15 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.14 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.14 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.14 GB):   3%|▎         | 2/58 [00:00<00:02, 19.03it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=59.14 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.14 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.13 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.12 GB):   9%|▊         | 5/58 [00:00<00:02, 22.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.12 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.98it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.12 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.98it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.12 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.98it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=59.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 22.98it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.11 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.80it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.11 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.11 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.11 GB):  19%|█▉        | 11/58 [00:00<00:01, 23.80it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.11 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.18it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.10 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.18it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.10 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.18it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=59.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 24.18it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.55it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.55it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.09 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.55it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.09 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.07 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.89it/s]Capturing num tokens (num_tokens=960 avail_mem=59.08 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.89it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=59.08 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.89it/s]Capturing num tokens (num_tokens=832 avail_mem=59.08 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.89it/s]Capturing num tokens (num_tokens=832 avail_mem=59.08 GB):  41%|████▏     | 24/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=768 avail_mem=59.07 GB):  41%|████▏     | 24/58 [00:00<00:01, 26.15it/s]Capturing num tokens (num_tokens=704 avail_mem=59.07 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.15it/s]Capturing num tokens (num_tokens=640 avail_mem=59.07 GB):  41%|████▏     | 24/58 [00:01<00:01, 26.15it/s]Capturing num tokens (num_tokens=640 avail_mem=59.07 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.10it/s]Capturing num tokens (num_tokens=576 avail_mem=59.07 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.10it/s]

    Capturing num tokens (num_tokens=512 avail_mem=59.05 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.10it/s]Capturing num tokens (num_tokens=480 avail_mem=59.07 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.10it/s]Capturing num tokens (num_tokens=480 avail_mem=59.07 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.23it/s]Capturing num tokens (num_tokens=448 avail_mem=59.06 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.23it/s]Capturing num tokens (num_tokens=416 avail_mem=59.06 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.23it/s]

    Capturing num tokens (num_tokens=384 avail_mem=59.06 GB):  52%|█████▏    | 30/58 [00:01<00:01, 22.23it/s]Capturing num tokens (num_tokens=384 avail_mem=59.06 GB):  57%|█████▋    | 33/58 [00:01<00:01, 20.88it/s]Capturing num tokens (num_tokens=352 avail_mem=59.05 GB):  57%|█████▋    | 33/58 [00:01<00:01, 20.88it/s]Capturing num tokens (num_tokens=320 avail_mem=59.05 GB):  57%|█████▋    | 33/58 [00:01<00:01, 20.88it/s]Capturing num tokens (num_tokens=288 avail_mem=59.05 GB):  57%|█████▋    | 33/58 [00:01<00:01, 20.88it/s]

    Capturing num tokens (num_tokens=288 avail_mem=59.05 GB):  62%|██████▏   | 36/58 [00:01<00:01, 20.21it/s]Capturing num tokens (num_tokens=256 avail_mem=59.04 GB):  62%|██████▏   | 36/58 [00:01<00:01, 20.21it/s]Capturing num tokens (num_tokens=240 avail_mem=59.04 GB):  62%|██████▏   | 36/58 [00:01<00:01, 20.21it/s]Capturing num tokens (num_tokens=224 avail_mem=59.04 GB):  62%|██████▏   | 36/58 [00:01<00:01, 20.21it/s]Capturing num tokens (num_tokens=224 avail_mem=59.04 GB):  67%|██████▋   | 39/58 [00:01<00:00, 19.34it/s]Capturing num tokens (num_tokens=208 avail_mem=59.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 19.34it/s]

    Capturing num tokens (num_tokens=192 avail_mem=59.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 19.34it/s]Capturing num tokens (num_tokens=176 avail_mem=59.03 GB):  67%|██████▋   | 39/58 [00:01<00:00, 19.34it/s]Capturing num tokens (num_tokens=176 avail_mem=59.03 GB):  72%|███████▏  | 42/58 [00:01<00:00, 19.18it/s]Capturing num tokens (num_tokens=160 avail_mem=59.03 GB):  72%|███████▏  | 42/58 [00:01<00:00, 19.18it/s]Capturing num tokens (num_tokens=144 avail_mem=59.02 GB):  72%|███████▏  | 42/58 [00:01<00:00, 19.18it/s]

    Capturing num tokens (num_tokens=144 avail_mem=59.02 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.28it/s]Capturing num tokens (num_tokens=128 avail_mem=59.02 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.28it/s]Capturing num tokens (num_tokens=112 avail_mem=59.02 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.28it/s]Capturing num tokens (num_tokens=96 avail_mem=59.01 GB):  76%|███████▌  | 44/58 [00:02<00:00, 19.28it/s] Capturing num tokens (num_tokens=96 avail_mem=59.01 GB):  81%|████████  | 47/58 [00:02<00:00, 18.09it/s]Capturing num tokens (num_tokens=80 avail_mem=59.01 GB):  81%|████████  | 47/58 [00:02<00:00, 18.09it/s]

    Capturing num tokens (num_tokens=64 avail_mem=59.01 GB):  81%|████████  | 47/58 [00:02<00:00, 18.09it/s]Capturing num tokens (num_tokens=48 avail_mem=59.00 GB):  81%|████████  | 47/58 [00:02<00:00, 18.09it/s]Capturing num tokens (num_tokens=48 avail_mem=59.00 GB):  86%|████████▌ | 50/58 [00:02<00:00, 19.97it/s]Capturing num tokens (num_tokens=32 avail_mem=59.00 GB):  86%|████████▌ | 50/58 [00:02<00:00, 19.97it/s]Capturing num tokens (num_tokens=28 avail_mem=59.00 GB):  86%|████████▌ | 50/58 [00:02<00:00, 19.97it/s]Capturing num tokens (num_tokens=24 avail_mem=58.99 GB):  86%|████████▌ | 50/58 [00:02<00:00, 19.97it/s]Capturing num tokens (num_tokens=20 avail_mem=58.99 GB):  86%|████████▌ | 50/58 [00:02<00:00, 19.97it/s]Capturing num tokens (num_tokens=16 avail_mem=58.99 GB):  86%|████████▌ | 50/58 [00:02<00:00, 19.97it/s]Capturing num tokens (num_tokens=16 avail_mem=58.99 GB):  95%|█████████▍| 55/58 [00:02<00:00, 26.52it/s]Capturing num tokens (num_tokens=12 avail_mem=58.98 GB):  95%|█████████▍| 55/58 [00:02<00:00, 26.52it/s]Capturing num tokens (num_tokens=8 avail_mem=58.98 GB):  95%|█████████▍| 55/58 [00:02<00:00, 26.52it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=58.98 GB):  95%|█████████▍| 55/58 [00:02<00:00, 26.52it/s]Capturing num tokens (num_tokens=4 avail_mem=58.98 GB): 100%|██████████| 58/58 [00:02<00:00, 23.23it/s]


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
    Generated text:  Tony, and I live in Atlanta, Georgia. I want to help people get out of poverty. I've worked with a variety of organizations, including VIDA, the Nonprofit Leadership Development Program, the Partnership for 21st Century Skills, and the Georgia Cooperative Extension. I'm particularly passionate about the VIDA program and the connections that it provides. 
    
    My name is Tony and I have a master's in education. I'm a Texas native who moved to Atlanta to work at the VIDA program. I can speak Spanish and have a good command of English. I also have a passion for education and helping people get out of
    ===============================
    Prompt: The president of the United States is
    Generated text:  30 years older than 3/5 of the total population of the United States. If the population is currently 350 million, what is the largest whole number that is definitely a factor of the president's age? Express your answer as a whole number.
    To determine the president's age, we first need to set up an equation based on the given information. Let \( P \) be the president's age and \( S \) be the total population of the United States. According to the problem, the president is 30 years older than \(\frac{3}{5}\) of the population. This
    ===============================
    Prompt: The capital of France is
    Generated text:  ( ). A. Paris B. London C. Madrid D. Rome
    A. The capital of France is ().
    A. Paris
    B. London
    C. Madrid
    D. Rome
    Answer: A
    
    In the total wage of production workers, the labor cost is the most significant, accounting for ( ) of the total wage.
    A. 50% to 80%
    B. 80% to 90%
    C. 90% to 95%
    D. 95% to 100%
    Answer: A
    
    Which of the following is NOT one of the three
    ===============================
    Prompt: The future of AI is
    Generated text:  strong and it's moving at a fast pace. In this blog post, we'll discuss the current trends and developments in AI, as well as the challenges and opportunities that it presents.
    
    AI is the use of artificial intelligence, a type of computer technology that is designed to think, learn, and solve problems in ways that are more efficient and accurate than humans. The use of AI has had a significant impact on various industries, including finance, healthcare, transportation, and more.
    
    One of the key trends in AI is the rise of large-scale machine learning. This involves using massive amounts of data to train machine learning algorithms, which can then be


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character, such as "funny, witty, and always up for a good laugh"]. I enjoy [insert a short description of your character's interests, such as "reading, cooking, or playing sports"]. I'm always looking for new challenges and opportunities to grow and learn. What do you think makes you unique? I think I'm unique because I'm always looking for new ways to learn and grow,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French Parliament Building. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also a major economic center and a major center of politics and government. The city is known for its cuisine, fashion, and music. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly together. It is a city that is both beautiful and exciting, and is a must
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some potential trends include:
    
    1. Increased use of AI in healthcare: AI is already being used in medical diagnosis and treatment, but it has the potential to revolutionize the field. AI-powered diagnostic tools could improve accuracy and speed, while AI-powered treatments could lead to more personalized and effective care.
    
    2. AI in manufacturing: AI is already being used in manufacturing to optimize production processes and improve quality control. As AI technology continues to improve, we can expect to see even more widespread adoption in manufacturing.
    
    3. AI in finance:
    


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
    Generated text:  [insert name], and I am a [insert profession or role]. I enjoy [insert something that makes me happy] and I love [insert something I like that relates to the character's profession or role]. I am [insert any additional information about the character, such as their interests, hobbies, or any notable achievements].
    I'm glad you stopped by to meet me. As a [insert profession or role], I enjoy [insert something that makes me happy] and I love [insert something I like that relates to the character's profession or role]. I am [insert any additional information about the character, such as their interests, hobbies
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the capital city of France, located in the heart of the French countryside, known for its iconic landmarks such as the Eiffel Tower and Notre-Dame Cathedral, as well as its vibrant culture, cuisine, and annual festivals. The city is home to several world-renowned universities, including the University of Paris, and hosts numerous museums, art galleries, and concert venues. Paris is a popular tourist destination, drawing millions of visitors each year and making it one of the world's most visited cities. The city is also a cultural hub, known for its rich history and diverse population, which has helped to create a unique
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  incredibly exciting, with new breakthroughs and applications constantly being developed and tested. Here are some possible trends that are likely to shape AI in the coming years:
    
    1. Improved Natural Language Processing: The ability to understand and generate human language will be an essential part of AI. This will allow machines to communicate with humans more effectively and will help them to understand the context in which they are operating. We may see advancements in natural language understanding that allow machines to interpret human language in a way that is more nuanced and contextually relevant.
    
    2. Enhanced Cybersecurity: AI is likely to have a significant impact on cybersecurity in the coming years. AI-based


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

    'm

     a

    /an

     ______

    .

     I

     enjoy

     ______

    __.

     I

     have

     a

    /an

     ______

    __

    _.

     I

     value

     ______

    _.

     


    When

     someone

     asks

     me

     what

     I

     do

    ,

     I

     just

     say

     "

    As

     a

    /an

     ______

    ___

    ."

     


    Please

     describe

     your

     personality

     traits

    ,

     strengths

     and

     weaknesses

    ,

     and

     your

     goals

     in

     life

    .

     


    What

     inspires

     you

     to

     do

     what

     you

     do

    ?

     


    What

     do

     you

     like

     to

     do

     when

     you

    're

     not

     working

    ?

     


    What

     do

     you

     like

     to

     do

     when

     you

    're

     not

     at

     home

    ?

     


    What

     is

     your

     favorite

     hobby

    ?

     


    What

     is

     your

     favorite

     movie

    /

    album

     or

     TV

     show

    ?

     


    What

     is

     your

     favorite

     food

    ?

     What

     are

     some

     of

     your

     favorite

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     capital

     and

     largest

     city

     of

     France

    .

     It

     is

     located

     in

     the

     center

     of

     the

     country

    ,

     along

     the

     Se

    ine

     River

    .

     The

     city

     has

     a

     rich

     history

     and

     is

     known

     for

     its

     art

    ,

     culture

    ,

     and

     food

    .

     It

     is

     home

     to

     the

     E

    iff

    el

     Tower

     and

     many

     famous

     landmarks

    ,

     including

     the

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

     Paris

     is

     also

     a

     tourist

     destination

     and

     has

     a

     large

     population

     of

     French

     citizens

    .

     The

     city

     is

     also

     home

     to

     a

     diverse

     population

     and

     is

     known

     for

     its

     multicultural

    ism

    .

     As

     of

     

    2

    0

    2

    3

    ,

     Paris

     had

     a

     population

     of

     around

     

    2

    .

    
    
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

     and

     robotics

    ,

     as

     well

     as

     increased

     focus

     on

     ethical

     considerations

     and

     regulatory

     compliance

    .

     AI

     is

     currently

     being

     used

     in

     a

     wide

     range

     of

     applications

    ,

     from

     self

    -driving

     cars

     and

     chat

    bots

     to

     healthcare

     and

     finance

    .

     In

     the

     future

    ,

     it

     is

     likely

     to

     be

     used

     in

     more

     complex

     and

     complex

     applications

    ,

     such

     as

     autonomous

     weapons

     and

     climate

     change

     predictions

    .

     Additionally

    ,

     there

     is

     potential

     for

     AI

     to

     be

     used

     in

     ways

     that

     are

     both

     beneficial

     and

     potentially

     harmful

    ,

     which

     could

     lead

     to

     further

     ethical

     considerations

    .

     As

     AI

     technology

     continues

     to

     advance

    ,

     it

     is

     likely

     to

     have

     a

     significant

     impact

     on

     our

    



```python
llm.shutdown()
```
