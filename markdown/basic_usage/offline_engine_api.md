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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.10it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.09it/s]


    2026-05-13 03:00:50,720 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 03:00:50] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:28,  4.71s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.12it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.12it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.76it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.76it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.76it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.76it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.76it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.76it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.76it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.76it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.76it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 13.85it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 13.85it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 13.85it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 13.85it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 13.85it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 13.85it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 13.85it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 13.85it/s]

    Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:05<00:01, 17.36it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:05<00:01, 17.36it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:05<00:01, 17.36it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:05<00:01, 17.36it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:05<00:01, 17.36it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:05<00:01, 17.36it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:05<00:01, 17.36it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:05<00:01, 17.36it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:05<00:01, 17.36it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:05<00:01, 17.36it/s]Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:05<00:01, 17.36it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 25.95it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 25.95it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 25.95it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 25.95it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 25.95it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 25.95it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 25.95it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 25.95it/s]

    Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 25.95it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 32.82it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 32.82it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 32.82it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 32.82it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 32.82it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.36it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   3%|▎         | 2/58 [00:00<00:02, 19.07it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.77it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.84it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  31%|███       | 18/58 [00:00<00:01, 35.84it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  31%|███       | 18/58 [00:00<00:01, 35.84it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 35.84it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  31%|███       | 18/58 [00:00<00:01, 35.84it/s]Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.17it/s]Capturing num tokens (num_tokens=832 avail_mem=74.35 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.17it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.17it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.17it/s]Capturing num tokens (num_tokens=640 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.17it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.17it/s]Capturing num tokens (num_tokens=576 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=480 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=448 avail_mem=74.34 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.11it/s]

    Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.11it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=352 avail_mem=74.33 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=256 avail_mem=74.32 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  57%|█████▋    | 33/58 [00:00<00:00, 42.73it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=208 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=192 avail_mem=74.31 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.67it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  66%|██████▌   | 38/58 [00:01<00:00, 43.67it/s]Capturing num tokens (num_tokens=160 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=144 avail_mem=74.30 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=128 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=112 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=96 avail_mem=74.29 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.20it/s] Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 44.20it/s]Capturing num tokens (num_tokens=80 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=64 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=48 avail_mem=74.28 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=32 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.47it/s]

    Capturing num tokens (num_tokens=28 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.47it/s]Capturing num tokens (num_tokens=24 avail_mem=74.27 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=20 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=16 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=12 avail_mem=74.26 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=8 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.59it/s] Capturing num tokens (num_tokens=4 avail_mem=74.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.59it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 45.14it/s]Capturing num tokens (num_tokens=4 avail_mem=74.25 GB): 100%|██████████| 58/58 [00:01<00:00, 39.97it/s]


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
    Generated text:  Chen Shi and I am 20 years old. I want to become an information worker and I plan to do this by learning how to code. How can I go about this?
    
    Creating a career in the Information Technology field can be a challenging task for a young person like me. However, there are several options to pursue this career path, including attending online courses, taking on internships, or pursuing a degree in Computer Science or Information Systems.
    
    Here are some steps you can take to increase your chances of becoming an Information Technology (IT) professional:
    
    1. Build your skillset: Start by building your technical skills by taking online courses
    ===============================
    Prompt: The president of the United States is
    Generated text:  the head of government and is the commander-in-chief of the United States Navy and the United States陸軍. As the commander-in-chief, the president is responsible for the foreign and domestic policy of the United States. He is also the first commander of the armed forces.
    Does this next sentence follow, given the above sentence? 
    
    The president can only take command of the armed forces of the United States.
    
    Available options:
    [a]. yes;
    [b]. it is not possible to tell;
    [c]. no;
    [a].
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris
    
    B. Lyon
    
    C. Bordeaux
    
    D. Toulouse
    
    To determine the capital of France, let's review the information provided:
    
    1. Paris is the capital of France.
    2. Bordeaux is the capital of France.
    3. Lyon is not the capital of France.
    4. Toulouse is not the capital of France.
    
    Since the question specifies that the capital of France is "Paris," and we have confirmed that Paris is indeed the capital, the correct answer is:
    
    A. Paris
    
    Therefore, the capital of France is \boxed{A}.
    ===============================
    Prompt: The future of AI is
    Generated text:  here, with the development of artificial intelligence (AI) on the rise. Artificial intelligence has advanced the world of technology, and now we can see its impact in areas like healthcare, transportation, and even sports. With the development of AI, the importance of data in AI is becoming more apparent. Data plays a significant role in the success of AI, and it is crucial to collect, process, and use data properly to ensure the accurate prediction of future outcomes. In this way, AI will play a more significant role in shaping the future of the world. 
    
    Despite its importance, many people are still hesitant to embrace AI. Many people believe


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [job title] and I'm always looking for ways to [job title] my skills and knowledge. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title] with [number of years] years of experience in [industry]. I'm a [job title]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a city with a rich history and culture. It is located on the Seine River and is home to many famous landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is known for its vibrant arts scene, delicious cuisine, and beautiful architecture. The city is also home to many international organizations and events, making it a popular destination for tourists and locals alike. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. Its reputation as a cultural and artistic capital is evident in the many museums,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries, including manufacturing, transportation, and healthcare. This will lead to increased efficiency, reduced costs, and improved productivity.
    
    2. Enhanced human-AI collaboration: As AI becomes more integrated into our daily lives, we can expect to see more human-AI collaboration. This will likely involve more complex tasks, such as decision-making and problem-solving
    


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
    Generated text:  [name], and I am a [field of study/occupation] with a strong passion for [interest or hobby]. I am excited to learn more about you and the world around us.
    
    That's a great start! Can you tell me more about what you enjoy doing outside of school and work? I'm curious to learn about you as a person. 
    
    Yes, I enjoy hiking, photography, and reading. I am constantly learning and growing as a person, and I am always eager to share my knowledge with others. I believe that learning is the key to personal growth and success, and I am always looking for new ways to expand
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Paris is the largest city in France and the 9th largest city in the world. It is a major cultural and economic center and one of the most visited cities in the world. The city is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is also home to many other world-renowned institutions such as the Louvre Museum, the Bibliothèque nationale de France, and the Musée d’Orsay. The city has a rich history and culture dating back to the 6th century and continues to be a significant center for
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  one that is rapidly evolving and will likely see many changes and advancements in the coming years. Some possible future trends in AI include:
    
    1. Increased automation and artificial intelligence in manufacturing: AI is becoming more prevalent in manufacturing, with robots and other intelligent systems being used to increase efficiency and accuracy. This trend will likely continue as companies invest more in AI to increase productivity and reduce costs.
    
    2. Improved autonomous vehicles: Autonomous vehicles will continue to evolve and become more sophisticated, with the ability to communicate with other vehicles and interact with the environment. This trend will likely lead to an increase in autonomous driving, with cars becoming more self-driving in the future


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

    ],

     and

     I

     am

     a

     [

    年龄段

    ]

     who

     grew

     up

     in

     [

    specific

     place

    ]

     and

     have

     always

     been

     an

     avid

     [

    specific

     interest

    ].

     I

     started

     my

     career

     in

     [

    specific

     field

    ]

     and

     have

     hon

    ed

     my

     skills

     through

     a

     combination

     of

     [

    training

    ,

     experience

    ,

     etc

    .

    ].

     
    


    What

     makes

     you

     unique

     and

     what

     do

     you

     like

     to

     do

     when

     you

    're

     not

     writing

     stories

     or

     giving

     interviews

    ?

     
    


    I

     love

     spending

     time

     outdoors

    ,

     whether

     it

    's

     hiking

    ,

     playing

     sports

    ,

     or

     simply

     walking

     dogs

    .

     I

     am

     also

     a

     huge

     advocate

     for

     [

    specific

     cause

    /s

    it

    uation

    ],

     and

     always

     look

     to

     contribute

     to

     causes

     I

     believe

     in

    .

     
    


    Lastly

    ,

     I

     am

     a

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     


    Note

    :

     The

     phrase

     "

    The

     capital

     of

     France

    "

     is

     a

     general

     term

     that

     can

     refer

     to

     any

     city

     located

     in

     the

     country

     of

     France

    .

     This

     statement

     may

     apply

     to

     other

     cities

     in

     France

    ,

     as

     well

    .

     However

    ,

     it

    's

     important

     to

     note

     that

     Paris

     is

     typically

     referred

     to

     as

     the

     "

    capital

    "

     of

     France

     in

     official

     contexts

    .

     


    For

     example

    :


    -

     The

     capital

     of

     Greece

     is

     Athens

    .


    -

     The

     capital

     of

     Mexico

     is

     Mexico

     City

    .


    -

     The

     capital

     of

     Switzerland

     is

     Geneva

    .

     


    -

     The

     capital

     of

     Italy

     is

     Rome

    .

     


    -

     The

     capital

     of

     Portugal

     is

     Lisbon

    .

     


    -

     The

     capital

     of

     Belgium

     is

     Brussels

    .

     


    -

     The

     capital

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     highly

     complex

     and

     rapidly

     changing

    ,

     with

     the

     ability

     to

     create

     and

     utilize

     AI

     capable

     of

     making

     decisions

     and

     decisions

     that

     make

     choices

     based

     on

     ethical

     and

     moral

     principles

    .

     Some

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Personal

    ization

    :

     AI

     will

     become

     more

     personalized

    ,

     with

     machines

     being

     able

     to

     adapt

     to

     user

     preferences

     and

     tailor

     their

     responses

     to

     meet

     specific

     needs

    .
    


    2

    .

     Aug

    mented

     intelligence

    :

     AI

     will

     continue

     to

     enhance

     our

     understanding

     of

     the

     world

     around

     us

    ,

     allowing

     machines

     to

     better

     process

     and

     analyze

     information

     and

     data

    .
    


    3

    .

     Rob

    otic

     automation

    :

     AI

     will

     become

     more

     integrated

     into

     human

     workplaces

    ,

     with

     robots

     becoming

     increasingly

     capable

     of

     performing

     tasks

     that

     were

     previously

     done

     by

    



```python
llm.shutdown()
```
