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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.99it/s]


    2026-05-08 03:54:10,129 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 03:54:10] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:17,  4.52s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:36,  1.44it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.28it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.28it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.61it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.61it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.47it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.47it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.47it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.47it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.47it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.47it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.47it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.47it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.47it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.47it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.39it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.39it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.39it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.39it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.39it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.39it/s]

    Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.39it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.39it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.39it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 28.97it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 28.97it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.03it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=68.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.68 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.68 GB):   3%|▎         | 2/58 [00:00<00:03, 18.45it/s]Capturing num tokens (num_tokens=7168 avail_mem=68.67 GB):   3%|▎         | 2/58 [00:00<00:03, 18.45it/s]Capturing num tokens (num_tokens=6656 avail_mem=68.67 GB):   3%|▎         | 2/58 [00:00<00:03, 18.45it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=68.67 GB):   7%|▋         | 4/58 [00:00<00:03, 15.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=68.67 GB):   7%|▋         | 4/58 [00:00<00:03, 15.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=68.66 GB):   7%|▋         | 4/58 [00:00<00:03, 15.68it/s]Capturing num tokens (num_tokens=5632 avail_mem=68.66 GB):  10%|█         | 6/58 [00:00<00:03, 14.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=68.65 GB):  10%|█         | 6/58 [00:00<00:03, 14.09it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=68.65 GB):  10%|█         | 6/58 [00:00<00:03, 14.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=68.65 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.65 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=3840 avail_mem=68.65 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=3584 avail_mem=68.64 GB):  14%|█▍        | 8/58 [00:00<00:03, 14.07it/s]Capturing num tokens (num_tokens=3584 avail_mem=68.64 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=68.64 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.43it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=68.64 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=2816 avail_mem=68.64 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=2560 avail_mem=68.63 GB):  19%|█▉        | 11/58 [00:00<00:02, 17.43it/s]Capturing num tokens (num_tokens=2560 avail_mem=68.63 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.43it/s]Capturing num tokens (num_tokens=2304 avail_mem=68.63 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.62 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.43it/s]Capturing num tokens (num_tokens=1792 avail_mem=68.62 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.61 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.43it/s]Capturing num tokens (num_tokens=1280 avail_mem=68.61 GB):  26%|██▌       | 15/58 [00:00<00:01, 22.43it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=68.61 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=1024 avail_mem=68.59 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=960 avail_mem=68.61 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.20it/s] Capturing num tokens (num_tokens=896 avail_mem=68.60 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=832 avail_mem=68.60 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.20it/s]Capturing num tokens (num_tokens=768 avail_mem=68.60 GB):  34%|███▍      | 20/58 [00:01<00:01, 28.20it/s]Capturing num tokens (num_tokens=768 avail_mem=68.60 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.57it/s]Capturing num tokens (num_tokens=704 avail_mem=68.59 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.57it/s]Capturing num tokens (num_tokens=640 avail_mem=68.59 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.57it/s]Capturing num tokens (num_tokens=576 avail_mem=68.59 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.57it/s]Capturing num tokens (num_tokens=512 avail_mem=68.57 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.57it/s]

    Capturing num tokens (num_tokens=512 avail_mem=68.57 GB):  50%|█████     | 29/58 [00:01<00:00, 34.37it/s]Capturing num tokens (num_tokens=480 avail_mem=68.59 GB):  50%|█████     | 29/58 [00:01<00:00, 34.37it/s]Capturing num tokens (num_tokens=448 avail_mem=68.59 GB):  50%|█████     | 29/58 [00:01<00:00, 34.37it/s]Capturing num tokens (num_tokens=416 avail_mem=68.59 GB):  50%|█████     | 29/58 [00:01<00:00, 34.37it/s]Capturing num tokens (num_tokens=384 avail_mem=68.58 GB):  50%|█████     | 29/58 [00:01<00:00, 34.37it/s]Capturing num tokens (num_tokens=352 avail_mem=68.58 GB):  50%|█████     | 29/58 [00:01<00:00, 34.37it/s]Capturing num tokens (num_tokens=352 avail_mem=68.58 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=320 avail_mem=68.57 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=288 avail_mem=68.57 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=256 avail_mem=68.57 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.78it/s]

    Capturing num tokens (num_tokens=240 avail_mem=68.56 GB):  59%|█████▊    | 34/58 [00:01<00:00, 35.78it/s]Capturing num tokens (num_tokens=240 avail_mem=68.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=224 avail_mem=68.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=208 avail_mem=68.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=192 avail_mem=68.56 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=176 avail_mem=68.55 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=160 avail_mem=68.55 GB):  66%|██████▌   | 38/58 [00:01<00:00, 34.91it/s]Capturing num tokens (num_tokens=160 avail_mem=68.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.84it/s]Capturing num tokens (num_tokens=144 avail_mem=68.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.84it/s]Capturing num tokens (num_tokens=128 avail_mem=68.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.84it/s]Capturing num tokens (num_tokens=112 avail_mem=68.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.84it/s]

    Capturing num tokens (num_tokens=96 avail_mem=68.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.84it/s] Capturing num tokens (num_tokens=80 avail_mem=68.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 37.84it/s]Capturing num tokens (num_tokens=80 avail_mem=68.53 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.18it/s]Capturing num tokens (num_tokens=64 avail_mem=68.53 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.18it/s]Capturing num tokens (num_tokens=48 avail_mem=68.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.18it/s]Capturing num tokens (num_tokens=32 avail_mem=68.49 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.18it/s]Capturing num tokens (num_tokens=28 avail_mem=68.48 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.18it/s]

    Capturing num tokens (num_tokens=24 avail_mem=68.24 GB):  83%|████████▎ | 48/58 [00:01<00:00, 40.18it/s]Capturing num tokens (num_tokens=24 avail_mem=68.24 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=20 avail_mem=68.25 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=16 avail_mem=68.47 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=12 avail_mem=68.46 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=8 avail_mem=68.45 GB):  91%|█████████▏| 53/58 [00:01<00:00, 33.42it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=68.45 GB):  98%|█████████▊| 57/58 [00:01<00:00, 30.33it/s]Capturing num tokens (num_tokens=4 avail_mem=68.45 GB):  98%|█████████▊| 57/58 [00:01<00:00, 30.33it/s]Capturing num tokens (num_tokens=4 avail_mem=68.45 GB): 100%|██████████| 58/58 [00:02<00:00, 28.85it/s]


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
    Generated text:  Jimmy and I have been a teacher since 1998. I taught at the University of South Florida, and for 8 years I taught at the University of North Carolina at Chapel Hill. I have also been a teacher at the Community College of Virginia. I was nominated for the 2015 Southern Regional Education Board Teacher of the Year Award. I am currently in the 7th grade and teaching 2nd grade. My interest in teaching and learning has always been strong since I was a child. I am a very supportive teacher and would love to work with a diverse group of students and families. I would love
    ===============================
    Prompt: The president of the United States is
    Generated text:  a federal officer of the United States and has the power to appoint and remove federal judges. There are no permanent positions, and appointments are made on an as needed basis.
    The president nominates the Secretary of Education and appoints the Secretary of Health and Human Services.
    President John F. Kennedy signed into law the Social Security Act of 1935 which created Social Security (later renamed Social Security Act of 1938) which was designed to provide the government with a source of revenue that would be available to fill government gaps, to expand benefits for elderly and young adults, to provide welfare, and to help prevent the creation
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris. Paris is located on the banks of the Seine River, the longest river in the world, and has been home to many famous people including Napoleon Bonaparte. It is also famous for its museums and monuments, and is home to a large number of international organizations.
    The area around Paris is also popular for its culture, such as the shopping district of the Old Quarter, the district of the Louvre, and the district of the Marais. It is also well-known for its famous cuisine and the famous restaurants and cafes.
    Paris is a city that is full of life, with a mix of different cultures and
    ===============================
    Prompt: The future of AI is
    Generated text:  expected to be “fully autonomous”, with robots and artificial intelligence emerging as the dominant form of technology in society. These technologies will change the way that we live, work and play, and they are expected to reduce human dependence on humans. While these technologies have the potential to bring many benefits, they also come with significant risks and challenges. One of the main risks is that the rise of these technologies could lead to job loss and a reduction in the quality of life for people who are not able to adapt to the new technologies.
    The rise of AI and robots is expected to have a significant impact on the job market in the near future. In


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive, enthusiastic, or neutral statement about yourself]. I enjoy [insert a short, positive, enthusiastic, or neutral statement about your hobbies or interests]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short, positive, enthusiastic, or neutral statement about your favorite hobby or activity]. I'm always looking for new ways to challenge myself
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a bustling metropolis with a rich history and a diverse population of over 10 million people. The city is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also home to many famous museums, including the Louvre and the Musée d'Orsay. The city is a popular tourist destination and a major economic center in Europe. It is also known for its cuisine, including French cuisine, which is famous for its rich flavors and use of fresh ingredients. Paris is a city that is constantly evolving
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and it has the potential to become even more advanced in the future. AI-powered diagnostic tools, such as machine learning algorithms, could help doctors make more accurate diagnoses and provide better treatment options for patients.
    
    2. Increased Use of AI in Manufacturing: AI is already being used in manufacturing to optimize production processes and improve quality control.
    


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
    Generated text:  [insert character's name], and I'm a [insert fictional profession]. I enjoy [insert interest in the field]. I've always been fascinated by [insert an interesting fact or topic] and have always wanted to learn more about it. My love for learning has led me to pursue [insert a career path or goal], and I'm excited to bring my passion for [insert a career field or topic] to my work. What kind of work do you do in your field, and why do you want to join us? [insert work details, such as "I work in [insert a field name] and spend my days [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France and the 16th most populous city in the world. It is located in the northwestern part of the country on the banks of the Seine River. The city is known for its rich history, art, and culture, as well as its bustling streets and vibrant food scene. Paris is home to iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre Dame Cathedral, and hosts numerous cultural events and festivals throughout the year. The city is also known for its cuisine, with dishes like croissants and duck dishes being popular throughout France. Paris is a popular
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly evolving and there are several potential trends that are likely to shape the development of this technology in the coming years:
    
    1. Increased use of AI in healthcare: AI-powered medical devices and systems are being developed to improve patient outcomes and reduce healthcare costs. AI will become more widely used in healthcare to assist doctors in diagnosing diseases, providing personalized treatment plans, and analyzing medical records.
    
    2. Improved efficiency and productivity: AI will be used to optimize production processes, improve customer service, and enhance logistics and supply chain management. AI-powered chatbots and virtual assistants will be more prevalent in customer support and customer service roles.
    
    3. Enhanced personalization


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

     profession

     or

     occupation

    ]

     from

     [

    Your

     country

     of

     origin

    ].

     I

    'm

     really

     excited

     to

     start

     my

     new

     adventure

     and

     [

    brief

    ly

     describe

     one

     or

     two

     achievements

     or

     experiences

     that

     have

     made

     you

     what

     you

     are

     today

    ].

     I

    'm

     here

     to

     [

    the

     purpose

     of

     your

     story

    ,

     perhaps

     to

     share

     a

     personal

     story

     or

     announce

     your

     arrival

    ].

     Thank

     you

     for

     asking

    !

     

    🌟

    ✨

    
    


    ---
    


    **

    This

     is

     just

     a

     placeholder

     for

     your

     reference

    ,

     I

     don

    't

     actually

     have a

     specific

     profession

     or

     country

    .

     I

    'll

     keep

     it

     neutral

     so

     no

     one

     will

     feel

     judged

    .

     Let

     me

     know

     if

     you

     want

     me

     to

     change

     this

     for

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     Lo

    ire

     Valley

     region

     of

     France

    .

     It

     is

     the

     largest

     city

     in

     the

     country

     and

     the

     second

     most

     populous

    ,

     with

     a

     population

     of

     over

     

    7

     million

     people

    .

     Paris

     is

     renowned

     for

     its

     iconic

     landmarks

     such

     as

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

     and

     Notre

    -D

    ame

     Cathedral

    ,

     and

     is

     known

     for

     its

     rich

     history

    ,

     culture

    ,

     and

     cuisine

    .

     The

     city

     also

     hosts

     numerous

     cultural

     and

     sporting

     events

     throughout

     the

     year

    ,

     and

     is

     considered

     one

     of

     the

     most

     important

     and

     vibrant

     cities

     in

     the

     world

    .

     Paris

     is

     one

     of

     the

     top

     destinations

     for

     tourists

    ,

     with

     its

     romantic

     architecture

    ,

     artistic

     scenes

    ,

     and

     vibrant

     nightlife

     attracting

     visitors

     from

     all

     over

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     rapidly

     evolving

     with

     new

     technologies

     emerging

     at

     an

     unprecedented

     pace

    .

     Here

     are

     some

     potential

     trends

     in

     AI

     that

     are

     shaping

     its

     future

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     AI

    :

     As

     more

     people

     become

     concerned

     about

     AI

    's

     impact

     on

     society

    ,

     there

     is

     growing

     recognition

     of

     the

     need

     to

     consider

     ethical

     implications

     of

     AI

    .

     This

     includes

     things

     like

     bias

     in

     algorithms

    ,

     the

     potential

     for

     AI

     to

     cause

     harm

    ,

     and

     the

     need

     to

     ensure

     that

     AI

     is

     developed

     and

     used

     in

     ways

     that

     are

     transparent

    ,

     accountable

    ,

     and

     equitable

    .
    


    2

    .

     Rise

     of

     AI

    -driven

     smart

     cities

    :

     As

     more

     and

     more

     cities

     adopt

     AI

     technologies

     to

     improve

     their

     efficiency

     and

     reduce

     waste

    ,

     it

     is

     likely

     that

    



```python
llm.shutdown()
```
