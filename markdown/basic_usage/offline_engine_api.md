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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.88it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.87it/s]


    2026-05-13 10:00:41,403 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 10:00:41] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:01,  4.23s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.60it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.60it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.60it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]

    Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.71it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:01, 19.31it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:01, 19.31it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:01, 19.31it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:01, 19.31it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:01, 19.31it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:01, 19.31it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:01, 19.31it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:04<00:01, 19.31it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:04<00:01, 19.31it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:04<00:01, 19.31it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 27.39it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 27.39it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 27.39it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 27.39it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 27.39it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]

    Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 27.39it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 37.41it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 37.41it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 37.41it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.65 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.62 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 21.50it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.18 GB):   9%|▊         | 5/58 [00:00<00:02, 21.50it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 21.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.17 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.82it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.17 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.82it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.16 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.82it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.16 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.82it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.16 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.53it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.15 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.53it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.15 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.53it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.15 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.53it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.15 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.14 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.14 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.92it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.14 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.14 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.92it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.13 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.13 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.92it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.11 GB):  28%|██▊       | 16/58 [00:00<00:01, 32.92it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=74.11 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.51it/s]Capturing num tokens (num_tokens=960 avail_mem=74.13 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.51it/s] Capturing num tokens (num_tokens=896 avail_mem=74.12 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.51it/s]Capturing num tokens (num_tokens=832 avail_mem=74.12 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.51it/s]Capturing num tokens (num_tokens=768 avail_mem=73.96 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.51it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  36%|███▌      | 21/58 [00:00<00:01, 35.51it/s]Capturing num tokens (num_tokens=704 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=640 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=576 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=512 avail_mem=73.93 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.90it/s]Capturing num tokens (num_tokens=480 avail_mem=73.95 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.90it/s]

    Capturing num tokens (num_tokens=480 avail_mem=73.95 GB):  52%|█████▏    | 30/58 [00:00<00:00, 38.17it/s]Capturing num tokens (num_tokens=448 avail_mem=73.95 GB):  52%|█████▏    | 30/58 [00:00<00:00, 38.17it/s]Capturing num tokens (num_tokens=416 avail_mem=73.95 GB):  52%|█████▏    | 30/58 [00:00<00:00, 38.17it/s]Capturing num tokens (num_tokens=384 avail_mem=73.94 GB):  52%|█████▏    | 30/58 [00:00<00:00, 38.17it/s]Capturing num tokens (num_tokens=352 avail_mem=73.94 GB):  52%|█████▏    | 30/58 [00:00<00:00, 38.17it/s]Capturing num tokens (num_tokens=320 avail_mem=73.93 GB):  52%|█████▏    | 30/58 [00:00<00:00, 38.17it/s]Capturing num tokens (num_tokens=320 avail_mem=73.93 GB):  60%|██████    | 35/58 [00:01<00:00, 40.19it/s]Capturing num tokens (num_tokens=288 avail_mem=73.93 GB):  60%|██████    | 35/58 [00:01<00:00, 40.19it/s]Capturing num tokens (num_tokens=256 avail_mem=73.93 GB):  60%|██████    | 35/58 [00:01<00:00, 40.19it/s]Capturing num tokens (num_tokens=240 avail_mem=73.92 GB):  60%|██████    | 35/58 [00:01<00:00, 40.19it/s]Capturing num tokens (num_tokens=224 avail_mem=73.92 GB):  60%|██████    | 35/58 [00:01<00:00, 40.19it/s]Capturing num tokens (num_tokens=208 avail_mem=73.92 GB):  60%|██████    | 35/58 [00:01<00:00, 40.19it/s]

    Capturing num tokens (num_tokens=208 avail_mem=73.92 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=192 avail_mem=73.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=176 avail_mem=73.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=160 avail_mem=73.91 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=144 avail_mem=73.90 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=128 avail_mem=73.90 GB):  69%|██████▉   | 40/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=128 avail_mem=73.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=112 avail_mem=73.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=96 avail_mem=73.90 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.06it/s] Capturing num tokens (num_tokens=80 avail_mem=73.89 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=64 avail_mem=73.89 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.06it/s]Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  78%|███████▊  | 45/58 [00:01<00:00, 43.06it/s]

    Capturing num tokens (num_tokens=48 avail_mem=73.89 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=32 avail_mem=73.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=28 avail_mem=73.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=24 avail_mem=73.88 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=20 avail_mem=73.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=16 avail_mem=73.87 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=16 avail_mem=73.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=12 avail_mem=73.87 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=8 avail_mem=73.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.99it/s] Capturing num tokens (num_tokens=4 avail_mem=73.86 GB):  95%|█████████▍| 55/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=4 avail_mem=73.86 GB): 100%|██████████| 58/58 [00:01<00:00, 38.16it/s]


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
    Generated text:  Daniel and I am the director of the Broadband and Innovation Forum, which is a joint initiative of the Broadband Broadband Association and the Broadband Innovation Association.
    The Broadband Innovation Forum (BIF) is the only dedicated forum dedicated to broadband innovation. It brings together the broadband industry’s leading minds, from research institutions to technology companies, to create a network of experts who can contribute to the future of broadband infrastructure.
    I am also the CEO of the Broadband and Innovation Forum. In addition, I am an advisor to many of the world’s leading broadband companies and major stakeholders in the broadband space.
    Currently, I serve on the
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a new job offer. The job offer has a start date of 2024. The president is considering two options: 
    
    1. The first option is to take the job offer and join the organization immediately, which would mean that the president would have to leave his current job, which he has been with the organization for over 20 years. 
    2. The second option is to start taking the job offer in 2023 and then leave the organization in 2025, which would mean that he would be able to work part-time and still maintain a high level of performance.
    
    If the president
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Moscow
    D. Tokyo
    Answer:
    
    A
    
    When using the PCD (Powerful CD) test paper method to evaluate myocardial ischemia in the elderly, which of the following tests is inappropriate?
    A. Carotid artery stenosis test
    B. Blood pressure measurement
    C. Electrocardiogram (ECG) and troponin level determination
    D. Serum creatinine level determination
    E. Thrombosis assessment
    Answer:
    
    D
    
    When a company's capital reserve is converted into capital, the difference between the amount of the capital reserve and the
    ===============================
    Prompt: The future of AI is
    Generated text:  not only about programming and machine learning, but also about how we engage with the world. It's about making decisions based on a wide range of factors, including but not limited to cultural, social, and economic influences. It's about being aware of our biases, and using AI tools to help us overcome them. It's also about being open to change and willing to adapt to new technologies and trends.
    Ultimately, the future of AI is a complex and ever-evolving landscape. As technology continues to advance, we will need to be agile and creative in our approach to using AI to improve the lives of people around the world. By embracing


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your profession or role]. I enjoy [insert a brief description of your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always up for a good challenge and love to explore new places and experiences. What's your favorite book or movie? I love [insert a short description of your favorite book or movie]. I'm always on
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Museum, and the French Parliament building. Paris is a cultural and historical center with a rich history dating back to the Roman Empire and the French Revolution. It is a major transportation hub and a major tourist destination. The city is known for its cuisine, fashion, and art, and is home to many world-renowned museums and galleries. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks of AI, there is likely to be a greater emphasis on ethical considerations. This could lead to more stringent regulations on AI development and deployment, as well as increased investment in research and development to ensure that AI is used in a responsible and beneficial way.
    
    2. Greater use of AI in healthcare: AI is already being used in a number of healthcare applications, from personalized medicine to disease diagnosis and treatment
    


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
    Generated text:  [Name], and I'm a [character] who has been around for [time] years. I've always been passionate about [occupation or hobby], and I'm always eager to learn new things and share my knowledge with others. Whether it's through writing, speaking, or whatever else I enjoy, I'm a person who loves to learn and grow. My personality is friendly, open-minded, and always looking for ways to help others. I'm a team player and always happy to contribute to any project or cause that inspires me. I believe in the power of collaboration and I'm always willing to help others who need it. Thank
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A. Correct
    B. Incorrect
    A. Correct
    Paris is the capital and largest city of France. It is known for its rich history, stunning architecture, and vibrant culture. It is also the center of the French language and is home to many of France's notable landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. As one of the world's most popular tourist destinations, Paris is a major hub for business, education, and entertainment. The French language is spoken by over 77 million people, making Paris a central hub for French culture and language in the world.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a variety of trends, including:
    
    1. Advancements in machine learning and deep learning: As technology continues to advance, we are likely to see even more sophisticated models that can process and analyze large amounts of data, leading to even more accurate predictions and better decision-making.
    
    2. Emergence of new applications: As AI technologies continue to evolve, we are likely to see new applications that capitalize on the power of AI, such as healthcare, education, and transportation.
    
    3. Integration with other technologies: As AI becomes more integrated into our daily lives, we are likely to see new applications that leverage the power of AI


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

    ],

     and

     I

     am

     a

     [

    Your

     Profession

    /

    Role

    ]

     at

     [

    Your

     Organization

    ],

     where

     I

    'm

     currently

     [

    Your

     Position

    ].

     I

     enjoy

     working

     with

     a

     wide

     range

     of

     clients

    ,

     and

     I

     bring

     a

     positive

     energy

     to

     all

     the

     work

     I

     do

    .

     I

    'm

     passionate

     about

     the

     growth

     and

     success

     of

     my

     clients

    ,

     and

     I

    'm

     dedicated

     to

     helping

     them

     achieve

     their

     best

     in

     life

    .
    


    I

    'm

     a

     compassionate

     and

     empath

    etic

     person

     who

     values

     honesty

    ,

     integrity

    ,

     and

     honesty

     in

     everything

     I

     do

    .

     I

    'm

     always

     ready

     to

     help

     and

     support

     my

     clients

    ,

     and

     I

    'm

     a

     true

     friend

     to

     those

     who

     rely

     on

     me

    .

     I

     believe

     in

     the

     power

     of

     self

    -care

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     and

     most

     populous

     city

     of

     France

    ,

     located

     on

     the

     Se

    ine

     River

     in

     the

     Mos

    elle

     region

     of

     northern

     France

    .

     It

     is

     known

     for

     its

     beautiful

     architecture

    ,

     rich

     history

    ,

     and

     vibrant

     culture

    ,

     and

     is

     an

     important

     city

     for

     commerce

    ,

     education

    ,

     and

     entertainment

    .

     It

     is

     also

     the

     home

     to

     some

     of

     the

     world

    's

     most

     famous

     museums

    ,

     including

     the

     Lou

    vre

     and

     the

     Grand

     Pal

    ais

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     a

     popular

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     annually

    .

     The

     city

     is

     home

     to

     more

     than

     

    7

     million

     inhabitants

    ,

     and

     is

     the

     third

    -largest

     city

     in

     the

     European

     Union

    .

     Its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     may

     involve

     many

     different

     developments

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

     landscape

     of

     AI

    :
    


    1

    .

     Autonomous

     vehicles

    :

     As

     the

     technology

     for

     autonomous

     vehicles

     continues

     to

     improve

    ,

     we

     may

     see

     more

     of

     these

     vehicles

     on

     the

     roads

    ,

     with

     machines

     that

     can

     drive

     safely

     and

     efficiently

    .

     Autonomous

     vehicles

     could

     potentially

     reduce

     traffic

     congestion

     and

     improve

     public

     safety

    .
    


    2

    .

     Language

     translation

     and

     interpretation

    :

     AI

     systems

     are

     already

     capable

     of

     translating

     between

     multiple

     languages

    ,

     but

     this

     technology

     has

     the

     potential

     to

     be

     even

     more

     advanced

     in

     the

     future

    .

     AI

     could

     be

     used

     to

     interpret

     speech

     and

     language

     more

     accurately

    ,

     improving

     communication

     and

     cooperation

    .
    


    3

    .

     Personal

    ized

     healthcare

    :

     AI

     could

     help

     improve

    



```python
llm.shutdown()
```
