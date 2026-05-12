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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.40it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.39it/s]


    2026-05-12 07:35:35,943 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 07:35:35] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:51,  5.12s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:41,  1.27it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:12,  3.82it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.15it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.15it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.15it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:02, 13.55it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 20.87it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 20.87it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 20.87it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 20.87it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 20.87it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 20.87it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 20.87it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 20.87it/s] 

    Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 20.87it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 20.87it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:05<00:00, 20.87it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:05<00:00, 29.13it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:05<00:00, 29.13it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.23 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.23 GB):   3%|▎         | 2/58 [00:00<00:03, 14.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.52 GB):   3%|▎         | 2/58 [00:00<00:03, 14.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.52 GB):   3%|▎         | 2/58 [00:00<00:03, 14.15it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.52 GB):   3%|▎         | 2/58 [00:00<00:03, 14.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.52 GB):   9%|▊         | 5/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.51 GB):   9%|▊         | 5/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.50 GB):   9%|▊         | 5/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.50 GB):   9%|▊         | 5/58 [00:00<00:02, 19.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.50 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.50 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.77it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=72.50 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.49 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.77it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.49 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.49 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.48 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.48 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.48 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.48 GB):  19%|█▉        | 11/58 [00:00<00:02, 22.99it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.48 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.47 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.77it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=72.47 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.47 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.47 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.45 GB):  28%|██▊       | 16/58 [00:00<00:01, 29.77it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.45 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.55it/s]Capturing num tokens (num_tokens=960 avail_mem=72.46 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.55it/s] Capturing num tokens (num_tokens=896 avail_mem=72.46 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.55it/s]Capturing num tokens (num_tokens=832 avail_mem=72.03 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.55it/s]Capturing num tokens (num_tokens=768 avail_mem=72.02 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.55it/s]Capturing num tokens (num_tokens=704 avail_mem=72.02 GB):  36%|███▌      | 21/58 [00:00<00:01, 33.55it/s]

    Capturing num tokens (num_tokens=704 avail_mem=72.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.28it/s]Capturing num tokens (num_tokens=640 avail_mem=72.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.28it/s]Capturing num tokens (num_tokens=576 avail_mem=72.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.28it/s]Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.28it/s]Capturing num tokens (num_tokens=480 avail_mem=72.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.28it/s]Capturing num tokens (num_tokens=448 avail_mem=72.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 37.28it/s]Capturing num tokens (num_tokens=448 avail_mem=72.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.81it/s]Capturing num tokens (num_tokens=416 avail_mem=72.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.81it/s]Capturing num tokens (num_tokens=384 avail_mem=72.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.81it/s]Capturing num tokens (num_tokens=352 avail_mem=72.00 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.81it/s]Capturing num tokens (num_tokens=320 avail_mem=72.00 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.81it/s]Capturing num tokens (num_tokens=288 avail_mem=72.00 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.81it/s]

    Capturing num tokens (num_tokens=288 avail_mem=72.00 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=240 avail_mem=71.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=224 avail_mem=71.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=208 avail_mem=71.98 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=192 avail_mem=71.98 GB):  62%|██████▏   | 36/58 [00:01<00:00, 40.02it/s]Capturing num tokens (num_tokens=192 avail_mem=71.98 GB):  71%|███████   | 41/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=176 avail_mem=71.98 GB):  71%|███████   | 41/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=160 avail_mem=71.98 GB):  71%|███████   | 41/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=144 avail_mem=71.97 GB):  71%|███████   | 41/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=128 avail_mem=71.97 GB):  71%|███████   | 41/58 [00:01<00:00, 40.57it/s]

    Capturing num tokens (num_tokens=112 avail_mem=71.97 GB):  71%|███████   | 41/58 [00:01<00:00, 40.57it/s]Capturing num tokens (num_tokens=112 avail_mem=71.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=96 avail_mem=71.97 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.69it/s] Capturing num tokens (num_tokens=80 avail_mem=71.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=64 avail_mem=71.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=48 avail_mem=71.95 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=32 avail_mem=71.95 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.69it/s]Capturing num tokens (num_tokens=32 avail_mem=71.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=28 avail_mem=71.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=20 avail_mem=71.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.09it/s]

    Capturing num tokens (num_tokens=16 avail_mem=71.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=12 avail_mem=71.93 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=12 avail_mem=71.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=8 avail_mem=71.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.57it/s] Capturing num tokens (num_tokens=4 avail_mem=71.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 42.57it/s]Capturing num tokens (num_tokens=4 avail_mem=71.93 GB): 100%|██████████| 58/58 [00:01<00:00, 36.11it/s]


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
    Generated text:  David and I am a researcher at the University of Oxford in the UK. I have been working on my doctoral research at the Department of Genetics at the University of Oxford, studying how the nervous system of humans and some other species is controlled by the brain. My aim is to understand how the brain controls the development of the nervous system during early embryogenesis. I have worked on this area for three years now, and I have just published a new paper in the Cell Press journal Development. I would like to thank the editor-in-chief for her interest in my work and the reviewers for their valuable feedback. I am keen to expand my research by
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the ___
    A. House of Representatives
    B. Senate
    C. President's Council
    D. Cabinet
    Answer:
    
    B
    
    When using the binomial distribution to estimate a population parameter, the conditions for the binomial distribution to be used are _____. ① The population is a discrete random variable with only two possible values; ② The population size N is large; ③ The sample size n is large; ④ The population is a normal distribution.
    A. ①②③
    B. ①②④
    C. ①③
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is in the country of France. What is the capital of France? Paris is the capital of France. It is located on the northern coast of the Iberian Peninsula, in the department of Seine-Saint-Denis, in the historic centre of the French Republic, near the Marne River. It is one of the world's most important cultural, academic, scientific, and artistic centers and is the most populous city in Europe, with a population of around 2. 4 million inhabitants. Paris is also the seat of the President of the French Republic. Paris has a unique and immense city center, with ten
    ===============================
    Prompt: The future of AI is
    Generated text:  intertwined with the advancements in nanotechnology, especially in the field of materials science. The current generation of AI systems have been driven by the need to solve complex problems in nanotechnology. Nanotechnology is the study of matter at the nanoscale, and this has led to the development of new materials and processes that can be used in various fields, including electronics, medicine, and energy.
    One of the key applications of nanotechnology in AI is in the field of electronics. Nanomaterials, such as graphene, have been found to have unique electronic properties that make them ideal for use in electronics. Researchers are exploring how these materials can be


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career and interests. Let's chat about [mention a specific topic or activity you enjoy doing]. I look forward to meeting you! [Name] [Company Name] [Job Title] [Company Address] [City, State, ZIP Code] [Phone Number] [Email Address] [LinkedIn Profile] [Twitter Profile] [Facebook Profile] [Website URL] [Company Website] [Company Social Media] [Company Blog] [Company Podcast] [Company YouTube Channel] [Company Instagram] [Company
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville-Marie" in French. It is the largest city in France and the third-largest city in the world by population. The city is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also a major center for business, finance, and education in Europe. Paris is a popular tourist destination and a major economic and cultural hub in France. The city is also home to many international organizations and institutions, including the European Parliament and the United Nations. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and context-aware AI systems.
    
    2. Enhanced ethical considerations: As AI becomes more integrated with human intelligence, there will be increased scrutiny of its ethical implications. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more integrated with human intelligence, it is likely
    


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
    Generated text:  [Your Name]. I'm a [Your Profession] with [Your Education, Previous Experience, or Career History] and have been [Your Career Path] for [Your Duration]. I enjoy [Your Profession], [Your Hobby, or Your Passion], and I'm a [Your Age, Gender, or Character Type]. I've always been interested in [Your Interests/Positions], and I'm always looking for [Your Goals or Achievements]. I'm also an [Your Personality Trait] and I'm always looking to learn new things. I'm always ready to help others and I'm willing to take risks, even if it means
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, often referred to as the City of Love and the City of Light. 
    
    This statement encapsulates the key aspects of Paris, including its historical significance, cultural attractions, and romantic associations with love. 
    
    To further expand on this statement, Paris has a rich history dating back over 700 years, which has shaped its unique character and appeal to visitors and locals alike. The city is home to countless landmarks and monuments, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, which are all major tourist attractions. 
    
    Paris also has a charming atmosphere and lively culture, with a vibrant nightlife and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be shaped by several key trends:
    
    1. Enhanced intelligence: AI is expected to continue to improve its ability to learn and make decisions, further broadening its applications. This includes improvements in natural language processing, image recognition, and speech recognition.
    
    2. Increased trust: As AI becomes more integrated into our daily lives, there is increasing pressure to build trust with users. This is likely to lead to more transparent and ethical AI development practices.
    
    3. Diversity and inclusion: AI is likely to become more diverse and inclusive, with more women, people with disabilities, and underrepresented groups in the field. This could lead to more fair and


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

     a

     [

    career

    ]

     in

     [

    field

     of

     expertise

    ]

    !

     My

     background

     is

     in

     [

    the

     specific

     area

     where

     I

     specialize

    ]

     and

     I

    've

     always

     been

     passionate

     about

     [

    the

     thing

     that

     gives

     me

     motivation

    ].

     I

    'm

     always

     looking

     for

     ways

     to

     [

    the

     thing

     I

     enjoy

     doing

    ]

     to

     bring

     joy

     to

     those

     around

     me

    .

     [

    If

     you

     have

     any

     specific

     questions

     about

     me

     or

     my

     career

     that

     I

     haven

    't

     addressed

     yet

    ,

     feel

     free

     to

     ask

     and

     I

    'll

     do

     my

     best

     to

     provide

     a

     brief

     introduction

    ].

     That

    's

     all

     I

     have

     for

     now

    !

     [

    End

     of

     intro

    ].

     Start

     with

     a

     more

     neutral

     self

    -int

    roduction

    .

     Hello

    ,

     my

     name

     is

    
    
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

     serves

     as

     the

     seat

     of

     government

    ,

     the

     capital

     of

     the

     country

    ,

     and

     the

     cultural

     and

     economic

     center

     of

     the

     country

    .

     It

     is

     also

     one

     of

     the

     oldest

     continuously

     inhabited

     cities

     in

     the

     world

    ,

     with

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

    .

     Paris

     is

     known

     for

     its

     stunning

     architecture

    ,

     vibrant

     culture

    ,

     and

     diverse

     cuisine

    ,

     and

     is

     a

     major

     center

     for

     music

    ,

     fashion

    ,

     and

     other

     arts

     and

     entertainment

    .

     The

     city

     is

     also

     famous

     for

     its

     annual

     May

     Day

     celebrations

    ,

     which

     are

     held

     in

     May

     and

     mark

     the

     beginning

     of

     summer

     in

     France

    .

     Paris

     is

     a

     popular

     tourist

     destination

    ,

     known

     for

     its

     iconic

     landmarks

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     dominated

     by

     three

     trends

    :

     autonomous

     AI

    ,

     where

     machines

     will

     be

     able

     to

     operate

     without

     human

     intervention

    ,

     and

     robotics

    ,

     where

     robots

     will

     become

     more

     autonomous

     and

     sophisticated

    .

     The

     integration

     of

     AI

     into

     everyday

     life

     is

     also

     expected

     to

     increase

    ,

     with

     AI

     systems

     becoming

     more

     prevalent

     in

     our

     homes

    ,

     workplaces

    ,

     and

     transportation

     systems

    .

     In

     addition

    ,

     there

     is

     a

     growing

     emphasis

     on

     ethical

     AI

    ,

     with

     concerns

     about

     the

     potential

     misuse

     of

     AI

     systems

     and

     the

     need

     for

     guidelines

     and

     regulations

     to

     ensure

     safe

     and

     ethical

     use

    .

     Finally

    ,

     there

     is

     an

     expectation

     that

     AI

     will

     continue

     to

     evolve

     and

     improve

    ,

     with

     new

     applications

     and

     technologies

     being

     developed

     every

     day

    .

     Overall

    ,

     the

     future

     of

    



```python
llm.shutdown()
```
