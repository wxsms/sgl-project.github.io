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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.80it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.79it/s]


    2026-05-07 00:54:11,339 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-07 00:54:11] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:34,  4.81s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:39,  1.35it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:39,  1.35it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:39,  1.35it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:12,  3.66it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:05<00:05,  7.72it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:05<00:05,  7.72it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:05<00:05,  7.72it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:05<00:05,  7.72it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:05<00:05,  7.72it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:05<00:05,  7.72it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:05<00:05,  7.72it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:05<00:05,  7.72it/s]Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:05<00:05,  7.72it/s]

    Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:05<00:02, 12.78it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 18.81it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 18.81it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 18.81it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 18.81it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 18.81it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 18.81it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 18.81it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 18.81it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:01, 18.81it/s]

    Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:05<00:01, 18.81it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 26.70it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 26.70it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 26.70it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 26.70it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 26.70it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 26.70it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 26.70it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 26.70it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 26.70it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:05<00:00, 26.70it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 35.30it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 35.30it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 35.30it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 35.30it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 35.30it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 35.30it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.35it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.67it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.67it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.67it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.67it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 21.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.31it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.31it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.31it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.31it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.31it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 29.24it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.47it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.47it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.47it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.47it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.47it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 35.47it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 39.71it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.59it/s]

    Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 42.59it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  57%|█████▋    | 33/58 [00:00<00:00, 45.08it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.26it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:00<00:00, 46.26it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.26it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.26it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.26it/s]

    Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  66%|██████▌   | 38/58 [00:01<00:00, 46.26it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.10it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.10it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.10it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.10it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.10it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  74%|███████▍  | 43/58 [00:01<00:00, 47.10it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]

    Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  83%|████████▎ | 48/58 [00:01<00:00, 46.82it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.51it/s] Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  91%|█████████▏| 53/58 [00:01<00:00, 45.51it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 45.65it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 40.80it/s]


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
    Generated text:  Adiea, and I’m a writer, designer, and graphic artist, working in all of the following areas: visual design, graphic design, writing, and design. I was born in România, but I live in Paris. I have been working in France since 2006.
    I have always been fascinated by the way the digital art and graphic arts have changed in the past few decades. I have been interested in digital art since the late 1990s, and I have been inspired by the rich history of graphic design. I am interested in the role of digital art and graphic design in the
    ===============================
    Prompt: The president of the United States is
    Generated text:  now trying to name his new presidential library. He wants to make sure that every person who visits the library can say they are a member of the Library's "community. " So he decides to name the library after the Thessalonians. He hopes that the name will inspire people to remember and acknowledge that he is the new president. 
    
    The Thessalonians were a group of Jewish believers who lived in the city of Thessalonica in what is now Thessaloniki, Greece. They were known for their piety and openness to new ideas. They were a unique group of people who set an example for others and
    ===============================
    Prompt: The capital of France is
    Generated text:  ____. 
    A. Paris
    B. Moscow
    C. London
    D. Hong Kong
    Answer:
    A
    
    The calculation result of the expression (-1)^{5} is ____
    A. 1
    B. -1
    C. -2
    D. 2
    Answer:
    B
    
    For the function f(x) = x^2 - 4x + 5, consider the following statements: ① The function is increasing on the interval [-2, 5]; ② The function has two distinct real roots; ③ The function has exactly one zero point; ④ The
    ===============================
    Prompt: The future of AI is
    Generated text:  not only in the knowledge and expertise of humans, but also in their physical and emotional conditions. This is where the future of AI lies. An AI is a machine that has the ability to perform tasks that humans have, but they are not capable of the psychological, emotional, or physical qualities that humans possess. One of the most important components of the future of AI is the ability of the machine to communicate with its user in a natural and intuitive way. This is because, without being able to communicate effectively, it is difficult for the machine to learn and adapt to new situations.
    It is also important to note that the future of AI is


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your job or profession]. I enjoy [insert a short description of your hobbies or interests]. I'm always looking for new experiences and learning new things. What's your favorite hobby or activity? I love [insert a short description of your favorite activity]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite book or movie? I love [insert a short description of your favorite book
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a city renowned for its historical significance, vibrant culture, and beautiful architecture. It is the largest city in France and the third-largest city in the world by population, with a population of over 2. 5 million people. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, and is a major center for art, literature, and music. It is also a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. Paris is a cultural and intellectual
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more prevalent in manufacturing, transportation, and other industries. Automation will likely lead to increased efficiency and productivity, but it will also lead to job displacement for some workers.
    
    2. AI ethics and privacy: As AI becomes more integrated into our lives, there will be increasing concerns about its ethical implications and potential privacy violations. There will be a need for regulations and guidelines to ensure that AI is used in a responsible and ethical manner.
    
    3
    


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
    Generated text:  [Name], and I am a [Type] who is [Age], [Gender] and [Occupation]. I believe that [Reason for being] is [Reason for being]. I am a [Type] because [Reason for being] is [Reason for being]. I hope to be a [Future Goal], and I strive to be a [Future Goal]. I am a [Type] because [Reason for being] is [Reason for being]. I am a [Type] because [Reason for being] is [Reason for being]. I believe that [Reason for being] is [Reason for being]. I am a [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the "City of Love" and "City of Light" due to its rich history and cultural richness. The city is famous for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, the Notre Dame Cathedral, and the Arc de Triomphe. Paris also has a diverse population, making it a vibrant and multicultural city, attracting tourists from around the world. It is the cultural, economic, and political center of France and plays an important role in French national identity. The city is often referred to as the "City of the Sun" due to its warm climate and Mediterranean influences. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  bright and promising, and there are many potential trends that could shape the field in the coming years. Here are some possible future trends in AI:
    
    1. Increased focus on ethical AI: As concerns about AI's impact on society and the environment grow, there is likely to be a greater emphasis on ethical AI principles, such as avoiding bias and transparency.
    
    2. AI development will become more diverse: As AI systems become more advanced, there is a growing need to ensure that they are designed and used in a way that is fair and unbiased.
    
    3. AI will become more localized: As AI systems become more complex, there will be a growing


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

     am

     a

     [

    background

     information

     about

     the

     character

    ,

     such

     as

     a

     profession

     or

     age

    ].

     I

     am

     [

    age

    ]

     years

     old

     and

     [

    gender

    ]

     -

     [

    gender

     identifier

    ].

     I

     currently

     live

     in

     [

    city

    ,

     country

    ].

     I

     have

     a

     love

     for

     [

    love

     interest

    ,

     hobby

    ,

     or

     sports

    ].

     I

     am

     always

     looking

     to

     learn

     new

     things

     and

     immer

    se

     myself

     in

     new

     cultures

    .

     I

     am

     passionate

     about

     [

    career

     goal

    ,

     hobby

    ,

     or

     outdoor

     activity

    ],

     and

     I

     am

     always

     trying

     to

     stay

     up

    -to

    -date

     with

     the

     latest

     trends

     and

     information

    .

     I

     am

     excited

     to

     see

     where

     my

     passion

     leads

     me

     in

     the

     future

    !

     What

     is

     your

     favorite

     hobby

     or

     activity

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    How

     is

     the

     currency

     in

     France

    ?


    The

     currency

     of

     France

     is

     the

     Euro

    .

     
    


    Which

     bank

     does

     the

     French

     Central

     Bank

     issue

     the

     French

     national

     currency

    ?


    The

     French

     Central

     Bank

     issues

     the

     French

     national

     currency

    ,

     the

     Euro

    .

     
    


    Is

     the

     French

     national

     currency

     the

     same

     as

     the

     Euro

    ?


    No

    ,

     the

     French

     national

     currency

     is

     the

     Euro

    .

     
    


    Do

     French

     citizens

     have

     the

     option

     to

     exchange

     their

     Euros

     for

     Dollars

    ?


    No

    ,

     French

     citizens

     do

     not

     have

     the

     option

     to

     exchange

     their

     Euros

     for

     Dollars

    .

     
    


    Is

     there

     a

     special

     bank

     in

     Paris

     called

     the

     "

    G

    are

     du

     Nord

    "

    ?


    Yes

    ,

     there

     is

     a

     special

     bank

     in

     Paris

     called

     the

     "

    G

    are

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     promising

    ,

     with

     many

     possibilities

     and

     possibilities

     for

     change

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     Efficiency

    :

     AI

     is

     becoming

     more

     efficient

     at

     tasks

     such

     as

     data

     analysis

    ,

     image

     and

     speech

     recognition

    ,

     and

     autonomous

     driving

    .

     As

     AI

     becomes

     more

     capable

    ,

     it

     will

     be

     able

     to

     handle

     more

     complex

     and

     time

    -consuming

     tasks

    .
    


    2

    .

     Personal

    ization

    :

     AI

     will

     become

     even

     more

     personalized

    ,

     with

     machines

     able

     to

     learn

     from

     user

     behavior

     and

     preferences

    ,

     offering

     tailored

     recommendations

     and

     services

    .
    


    3

    .

     Automation

    :

     AI

     will

     continue

     to

     automate

     many

     tasks

    ,

     such

     as

     manufacturing

    ,

     customer

     service

    ,

     and

     administrative

     work

    ,

     freeing

     up

     human

     workers

     to

     focus

     on

     more

     strategic

    ,

     creative

    



```python
llm.shutdown()
```
