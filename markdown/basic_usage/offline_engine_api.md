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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.34it/s]


    2026-04-15 21:10:34,466 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 21:10:34] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:23,  2.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:23,  2.52s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:23,  2.52s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:23,  2.52s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:27,  1.99it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.07it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.61it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.61it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.61it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.61it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.61it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.61it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 13.61it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:02<00:02, 13.61it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:02<00:02, 13.61it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:02<00:01, 20.97it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:02<00:01, 20.97it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:02<00:01, 20.97it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:02<00:01, 20.97it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:02<00:01, 20.97it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.97it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 28.81it/s]

    Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 28.81it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 35.26it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 35.26it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 41.28it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 41.28it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 41.28it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.23it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.79 GB):   2%|▏         | 1/58 [00:00<00:06,  9.00it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.76 GB):   2%|▏         | 1/58 [00:00<00:06,  9.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.75 GB):   2%|▏         | 1/58 [00:00<00:06,  9.00it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=71.75 GB):   2%|▏         | 1/58 [00:00<00:06,  9.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.75 GB):   7%|▋         | 4/58 [00:00<00:03, 17.70it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.75 GB):   7%|▋         | 4/58 [00:00<00:03, 17.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.75 GB):   7%|▋         | 4/58 [00:00<00:03, 17.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.75 GB):   7%|▋         | 4/58 [00:00<00:03, 17.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.55it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.74 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.55it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=71.74 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.55it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.62it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.62it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.62it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.72 GB):  19%|█▉        | 11/58 [00:00<00:01, 28.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.71 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.71 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.11it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=71.69 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.11it/s]Capturing num tokens (num_tokens=960 avail_mem=71.70 GB):  28%|██▊       | 16/58 [00:00<00:01, 35.11it/s] Capturing num tokens (num_tokens=960 avail_mem=71.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.91it/s]Capturing num tokens (num_tokens=896 avail_mem=71.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.91it/s]Capturing num tokens (num_tokens=832 avail_mem=71.62 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.91it/s]Capturing num tokens (num_tokens=768 avail_mem=71.16 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.91it/s]

    Capturing num tokens (num_tokens=704 avail_mem=71.16 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.91it/s]Capturing num tokens (num_tokens=640 avail_mem=71.15 GB):  38%|███▊      | 22/58 [00:00<00:00, 40.91it/s]Capturing num tokens (num_tokens=640 avail_mem=71.15 GB):  47%|████▋     | 27/58 [00:00<00:01, 26.48it/s]Capturing num tokens (num_tokens=576 avail_mem=71.15 GB):  47%|████▋     | 27/58 [00:00<00:01, 26.48it/s]

    Capturing num tokens (num_tokens=512 avail_mem=71.14 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.48it/s]Capturing num tokens (num_tokens=480 avail_mem=71.16 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.48it/s]Capturing num tokens (num_tokens=448 avail_mem=71.15 GB):  47%|████▋     | 27/58 [00:01<00:01, 26.48it/s]Capturing num tokens (num_tokens=448 avail_mem=71.15 GB):  53%|█████▎    | 31/58 [00:01<00:01, 21.75it/s]Capturing num tokens (num_tokens=416 avail_mem=71.15 GB):  53%|█████▎    | 31/58 [00:01<00:01, 21.75it/s]

    Capturing num tokens (num_tokens=384 avail_mem=71.15 GB):  53%|█████▎    | 31/58 [00:01<00:01, 21.75it/s]Capturing num tokens (num_tokens=352 avail_mem=71.14 GB):  53%|█████▎    | 31/58 [00:01<00:01, 21.75it/s]Capturing num tokens (num_tokens=352 avail_mem=71.14 GB):  59%|█████▊    | 34/58 [00:01<00:01, 20.86it/s]Capturing num tokens (num_tokens=320 avail_mem=71.14 GB):  59%|█████▊    | 34/58 [00:01<00:01, 20.86it/s]Capturing num tokens (num_tokens=288 avail_mem=71.14 GB):  59%|█████▊    | 34/58 [00:01<00:01, 20.86it/s]

    Capturing num tokens (num_tokens=256 avail_mem=71.13 GB):  59%|█████▊    | 34/58 [00:01<00:01, 20.86it/s]Capturing num tokens (num_tokens=256 avail_mem=71.13 GB):  64%|██████▍   | 37/58 [00:01<00:01, 21.00it/s]Capturing num tokens (num_tokens=240 avail_mem=71.13 GB):  64%|██████▍   | 37/58 [00:01<00:01, 21.00it/s]Capturing num tokens (num_tokens=224 avail_mem=71.13 GB):  64%|██████▍   | 37/58 [00:01<00:01, 21.00it/s]Capturing num tokens (num_tokens=208 avail_mem=71.12 GB):  64%|██████▍   | 37/58 [00:01<00:01, 21.00it/s]Capturing num tokens (num_tokens=192 avail_mem=71.12 GB):  64%|██████▍   | 37/58 [00:01<00:01, 21.00it/s]Capturing num tokens (num_tokens=192 avail_mem=71.12 GB):  71%|███████   | 41/58 [00:01<00:00, 24.08it/s]Capturing num tokens (num_tokens=176 avail_mem=71.12 GB):  71%|███████   | 41/58 [00:01<00:00, 24.08it/s]Capturing num tokens (num_tokens=160 avail_mem=71.12 GB):  71%|███████   | 41/58 [00:01<00:00, 24.08it/s]

    Capturing num tokens (num_tokens=144 avail_mem=71.11 GB):  71%|███████   | 41/58 [00:01<00:00, 24.08it/s]Capturing num tokens (num_tokens=128 avail_mem=71.11 GB):  71%|███████   | 41/58 [00:01<00:00, 24.08it/s]Capturing num tokens (num_tokens=128 avail_mem=71.11 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.23it/s]Capturing num tokens (num_tokens=112 avail_mem=71.11 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.23it/s]Capturing num tokens (num_tokens=96 avail_mem=71.10 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.23it/s] Capturing num tokens (num_tokens=80 avail_mem=71.10 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.23it/s]Capturing num tokens (num_tokens=64 avail_mem=71.09 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.23it/s]Capturing num tokens (num_tokens=48 avail_mem=71.09 GB):  78%|███████▊  | 45/58 [00:01<00:00, 27.23it/s]Capturing num tokens (num_tokens=48 avail_mem=71.09 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=32 avail_mem=71.09 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=28 avail_mem=71.08 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.40it/s]

    Capturing num tokens (num_tokens=24 avail_mem=71.08 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=20 avail_mem=71.07 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=16 avail_mem=71.07 GB):  86%|████████▌ | 50/58 [00:01<00:00, 31.40it/s]Capturing num tokens (num_tokens=16 avail_mem=71.07 GB):  95%|█████████▍| 55/58 [00:02<00:00, 34.95it/s]Capturing num tokens (num_tokens=12 avail_mem=71.07 GB):  95%|█████████▍| 55/58 [00:02<00:00, 34.95it/s]Capturing num tokens (num_tokens=8 avail_mem=71.07 GB):  95%|█████████▍| 55/58 [00:02<00:00, 34.95it/s] Capturing num tokens (num_tokens=4 avail_mem=71.06 GB):  95%|█████████▍| 55/58 [00:02<00:00, 34.95it/s]Capturing num tokens (num_tokens=4 avail_mem=71.06 GB): 100%|██████████| 58/58 [00:02<00:00, 28.11it/s]


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
    Generated text:  Ayyash. I am the youngest of the 10 kids in my family. I was born on July 15, 1992 in a village of Bangla-ka-ka-pahar, a small village of Bangla-ka-ka-pahar in Anantapur district, Andhra Pradesh, India. I was born with a chromosome abnormality (TRDS) of -21q21.3. I was born without a set of chromosomes. The only physical feature is a small and round face. The medical condition is called Trisomy-21 (also known as
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. Here are a few reasons why he is so important. The president makes sure everyone in the country is safe and that people are treated fairly. He also helps make important decisions. He is always looking for ways to help people, but he is always taking care of them. The president is there to make sure that everyone is treated fairly and that everyone is happy. The president is the leader of the country and is always trying to make sure that everyone is happy.
    What is the main point of the paragraph?
    A) A person who is important and helps make important decisions.
    B) A person who is the leader of
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. Berlin
    C. London
    D. Moscow
    A capital is a city that is the capital of a country. The capital of France is Paris. Therefore, the correct answer is:
    
    A. Paris
    B. Berlin
    C. London
    D. Moscow
    A. Paris (the capital of France)
    ===============================
    Prompt: The future of AI is
    Generated text:  just around the corner. As it’s often the case in science, the first prediction is almost always wrong. In the AI field, it’s been almost 15 years since AI technology was so good that we thought it would be used in a whole bunch of new amazing ways. Back then, the barriers to building AI were very low and the field was developing rapidly. The risk of AI was low and the potential rewards were big. This led to a lot of hype. But now, the AI hype is reaching the end of its life cycle and the companies that are building it will be looking to cut their losses. But how will


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your interests and passions. What can you tell me about yourself? I'm a [insert a short description of your personality or background]. I enjoy [insert a short description of your hobbies or interests]. What's your favorite hobby or activity? I love [insert a short description of your favorite hobby or activity]. What's your favorite book or movie? I love [insert a short description of your favorite book or movie]. What's your favorite place to go? I love [insert a short description of your favorite
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant nightlife and is a popular tourist destination. The city is home to many international organizations and is a major economic and cultural center in Europe. It is also known for its cuisine, including French cuisine, and is a popular destination for tourists and locals alike. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the city's vibrant culture. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that are expected to shape the development of AI in the coming years:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. In the future, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve risk management
    


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
    Generated text:  [insert character name], and I'm a/an [insert occupation or profession]. I'm a/an [insert age, nationality, and/or gender] [insert occupation or profession]. I grew up in [insert location] and I've always been passionate about [insert something specific]. I've always wanted to be a/an [insert occupation or profession], but I've always been intimidated by [insert occupation or profession]. So, I decided to give it a try and I'm here today to prove to myself and others that I can do anything I set my mind to. I'm a/an [insert occupation or profession] with a passion for
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the northwestern part of the country, on the River Seine.
    That's correct! Paris is the capital city of France, located on the River Seine in the center of the country. It's known for its rich history, stunning architecture, and vibrant culture. The city is home to iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also home to the French Quarter, a historic district with its own distinctive culture, as well as the city's museums, museums, and other attractions. With its beautiful architecture, rich history, and friendly people,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to evolve, with new technologies, trends, and applications emerging and developing rapidly. Here are some possible future trends in AI:
    
    1. Autonomous vehicles: As the number of vehicles on the road increases, we are likely to see more autonomous vehicles on the roads, with self-driving cars becoming more common. AI will play a crucial role in making these vehicles safer and more efficient.
    
    2. Personalized healthcare: AI will continue to play a key role in the healthcare industry, with more personalized treatment plans becoming possible. AI will help doctors diagnose and treat diseases more accurately and efficiently, and will also help patients manage chronic conditions.
    
    3


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

    'm

     a

     [

    Job

     Title

    ]

     from

     [

    Company

     Name

    ].

     I

    'm

     currently

     working

     on

     a

     project

     called

     [

    Project

     Name

    ],

     and

     I

    'm

     excited

     to

     work

     with

     you

    .

     If

     you

     need

     any

     help

     or

     have

     any

     questions

    ,

     please

     don

    't

     hesitate

     to

     ask

    .

     And

     if

     you

    're

     interested

     in

     my

     work

    ,

     you

     can

     contact

     me

     directly

     via

     [

    Contact

     Information

    ].

     I

    'm

     always

     here

     to

     help

    .

     [

    Name

    ]

     [

    Job

     Title

    ]

     

     [

    Company

     Name

    ]

     (

    Your

     name

    )

     [

    Company

     Address

    ]

     [

    City

    ,

     State

     ZIP

    ]

     (

    Your

     email

    )

     (

    Your

     phone

     number

    )

     [

    Name

    ]

     [

    Job

     Title

    ]

     [

    Company

     Name

    ]

     (

    Your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     center

     of

     the

     country

     and

     known

     as

     the

     City

     of

     Light

    .

     It

     is

     known

     for

     its

     beautiful

     architecture

    ,

     vibrant

     culture

    ,

     and

     historic

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

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     considered

     one

     of

     the

     most

     iconic

     cities

     in

     the

     world

    .

     It

     has

     a

     rich

     and

     diverse

     cultural

     and

     artistic

     heritage

    ,

     and

     is

     a

     center

     of

     international

     business

     and

     politics

    .

     Paris

    's

     most

     famous

     landmark

     is

     the

     E

    iff

    el

     Tower

    ,

     which

     is

     a

     UNESCO

     World

     Heritage

     Site

    .

     It

     is

     also

     home

     to

     the

     iconic

     Tour

     de

     France

     bicycle

     race

    .

     
    


    Paris

     has

     a

     complex

     history

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     combination

     of

     new

     technologies

    ,

     advances

     in

     data

     science

    ,

     and

     the

     increasing

     importance

     of

     ethical

     considerations

     in

     AI

     development

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

     Increased

     availability

     of

     AI

    -powered

     tools

     and

     services

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     can

     expect

     to

     see

     an

     increasing

     number

     of

     AI

    -powered

     tools

     and

     services

     available

     to

     people

    .

     These

     could

     include

     virtual

     assistants

    ,

     chat

    bots

    ,

     and

     intelligent

     transportation

     systems

    .
    


    2

    .

     Personal

    ized

     AI

    :

     AI

     is

     also

     likely

     to

     make

     significant

     advancements

     in

     the

     area

     of

     personalized

     AI

    ,

     where

     AI

     is

     used

     to

     analyze

     and

     extract

     insights

     from

     large

     amounts

     of

     data

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     healthcare

    ,

    



```python
llm.shutdown()
```
