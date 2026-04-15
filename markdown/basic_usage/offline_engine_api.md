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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.05it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.03it/s]


    2026-04-15 17:22:46,988 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 17:22:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:43,  2.86s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:30,  1.75it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:30,  1.75it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:30,  1.75it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:03<00:30,  1.75it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:03<00:30,  1.75it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:03<00:30,  1.75it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:03<00:30,  1.75it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:03<00:08,  5.39it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:03<00:08,  5.39it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:03<00:08,  5.39it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:03<00:08,  5.39it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:03<00:08,  5.39it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:03<00:08,  5.39it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.39it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.39it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.39it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.48it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.48it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.48it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.48it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.48it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.48it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.48it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.48it/s]

    Compiling num tokens (num_tokens=704):  31%|███       | 18/58 [00:03<00:03, 11.48it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.59it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.59it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.59it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.59it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.59it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.59it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.59it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.59it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]

    Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=192):  57%|█████▋    | 33/58 [00:03<00:00, 25.11it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:03<00:00, 33.10it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:03<00:00, 33.10it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:03<00:00, 33.10it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:03<00:00, 33.10it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:03<00:00, 33.10it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:03<00:00, 33.10it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:03<00:00, 33.10it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:03<00:00, 33.10it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:03<00:00, 33.10it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 40.70it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 40.70it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 40.70it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 40.70it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 40.70it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 40.70it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 40.70it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 40.70it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 40.70it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 40.70it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 50.13it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:02, 18.68it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.79it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.43it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.62 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.09it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.62 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.09it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.09it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.09it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.61 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.71it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.71it/s]Capturing num tokens (num_tokens=960 avail_mem=131.59 GB):  29%|██▉       | 17/58 [00:00<00:01, 27.71it/s] Capturing num tokens (num_tokens=960 avail_mem=131.59 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.22it/s]Capturing num tokens (num_tokens=896 avail_mem=131.59 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.22it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.22it/s]Capturing num tokens (num_tokens=768 avail_mem=131.58 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.22it/s]Capturing num tokens (num_tokens=704 avail_mem=131.58 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.22it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  38%|███▊      | 22/58 [00:00<00:01, 32.22it/s]

    Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.58it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.58it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.58it/s]Capturing num tokens (num_tokens=480 avail_mem=131.58 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.58it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.58it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  47%|████▋     | 27/58 [00:00<00:00, 35.58it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  55%|█████▌    | 32/58 [00:00<00:00, 37.93it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.93it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.93it/s]Capturing num tokens (num_tokens=288 avail_mem=131.56 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.93it/s]

    Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.93it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=224 avail_mem=131.55 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  64%|██████▍   | 37/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=160 avail_mem=131.54 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.76it/s]

    Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.76it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  72%|███████▏  | 42/58 [00:01<00:00, 40.76it/s] Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  81%|████████  | 47/58 [00:01<00:00, 41.37it/s]Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  81%|████████  | 47/58 [00:01<00:00, 41.37it/s]Capturing num tokens (num_tokens=64 avail_mem=131.52 GB):  81%|████████  | 47/58 [00:01<00:00, 41.37it/s]Capturing num tokens (num_tokens=48 avail_mem=131.52 GB):  81%|████████  | 47/58 [00:01<00:00, 41.37it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  81%|████████  | 47/58 [00:01<00:00, 41.37it/s]Capturing num tokens (num_tokens=28 avail_mem=131.51 GB):  81%|████████  | 47/58 [00:01<00:00, 41.37it/s]Capturing num tokens (num_tokens=28 avail_mem=131.51 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.88it/s]

    Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.88it/s]Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  90%|████████▉ | 52/58 [00:01<00:00, 41.88it/s] Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.63it/s]Capturing num tokens (num_tokens=4 avail_mem=131.49 GB):  98%|█████████▊| 57/58 [00:01<00:00, 42.63it/s]Capturing num tokens (num_tokens=4 avail_mem=131.49 GB): 100%|██████████| 58/58 [00:01<00:00, 36.74it/s]


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
    Generated text:  Ramesh. I am a 24-year-old software engineer with over 7 years of experience in software development. I currently work at a big tech company where I am a part-time employee and I have a few projects in progress with the company as well. I enjoy teaching people how to code and helping them to improve their coding skills. My passion is to help people get the most out of their programming skills by providing them with the best resources and support. I have a passion for learning and always strive to stay up-to-date with the latest technologies and programming languages. I'm excited to meet and collaborate with new people and share
    ===============================
    Prompt: The president of the United States is
    Generated text:  now trying to decide whether to hold another presidential election. There are 79 electoral votes to be distributed amongst the states, with 27 for the president, 27 for the vice president, 3 for the vice president of the Senate, 4 for the president of the Senate, 1 for the president of the Supreme Court, and 1 for the vice president of the Supreme Court. If 4 states have voted for the presidential candidate, 6 states have voted for the vice presidential candidate, and there are 18 states with no voting records, how many electoral votes does the president need to win?
    To
    ===============================
    Prompt: The capital of France is
    Generated text:  the city of Paris, located in the department of the Pas-de-Calais. Paris is a city of calm and gravity, and its streets are lined with classical monuments, museums, and art galleries. The city's most famous landmark is the Eiffel Tower, which is 324 meters high and attracts millions of tourists every year. Other landmarks in Paris include the Louvre Museum, the Place de la Concorde, the Notre-Dame Cathedral, and the Arc de Triomphe.
    Paris's art scene is known for its vibrant and diverse range of artistic expressions. The city is home to a diverse array of galleries,
    ===============================
    Prompt: The future of AI is
    Generated text:  expected to lead to the creation of a new form of artificial intelligence, which has the potential to transform various industries and improve people’s lives. This new form of AI is called smart AI, and it is designed to enhance the functionality and performance of existing AI systems, as well as to create entirely new applications. The concept of smart AI is also a significant advancement in the field of artificial intelligence, as it aims to improve efficiency and productivity, and to create more personalized and engaging experiences for users.
    One of the key advantages of smart AI is its ability to provide personalized and context-aware responses to users. This is achieved through machine learning algorithms that


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in the country and the seat of government and culture. The city is known for its rich history, beautiful architecture, and vibrant culture. Paris is also famous for its fashion industry and its annual Eiffel Tower celebration. The city is home to many famous landmarks such as the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. Paris is a popular tourist destination and a major economic center in France. It is also known for its cuisine, including French cuisine, and its wine industry. The city is home to many museums, theaters, and other cultural institutions. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations in a more natural way. This could lead to more efficient and effective use of AI in various fields, such as healthcare, finance, and transportation.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations, including issues such as bias, privacy
    


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
    Generated text:  [Name], and I'm an AI. What can you tell me about yourself? I am an AI language model, trained to understand and respond to natural language. I am here to assist and provide information to users who want to communicate with me. How can I assist you today? 
    
    For more detailed information about my abilities or capabilities, please let me know. I'll do my best to provide you with the most accurate and helpful response. If you have any specific questions or concerns, please let me know and I'll do my best to help you. 
    
    Remember, I am here to assist and provide information to users who want to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is known for its stunning architecture, rich history, and vibrant cultural scene. Paris is a popular tourist destination and a cultural melting pot, drawing visitors from around the world. The city has a long and storied history dating back to the Roman period, and has been the capital of France since the 12th century. Today, Paris is a major center for business, finance, fashion, and art, and attracts millions of visitors annually. Its iconic landmarks, such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum, are some of the most recognizable symbols of the city. Paris is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a variety of factors, including advancements in machine learning, quantum computing, and cognitive neuroscience. Some possible future trends in AI include:
    
    1. Increased automation: As AI becomes more capable of performing tasks that are typically done by humans, we are likely to see an increase in automation. For example, autonomous vehicles could significantly reduce accidents and improve efficiency in transportation.
    
    2. Personalized AI: As AI algorithms become more sophisticated, they may be able to learn more about individual users and provide personalized experiences. For example, chatbots could be designed to understand the emotional needs of users and respond in a way that feels more natural


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

    'm

     an

     experienced

     [

    insert

     skill

     or

     hobby

     here

    ].

     I

    've

     been

     working

     in

     the

     [

    insert

     industry

     here

    ]

     for

     [

    insert

     number

     of

     years

     here

    ].

     I

     enjoy

     [

    insert

     something

     that

     relates

     to

     your

     background

     or

     experience

     here

    ].

     I

    'm

     passionate

     about

     [

    insert

     something

     that

     relates

     to

     your

     background

     or

     experience

     here

    ].

     What

     can

     you

     tell

     me

     about

     yourself

    ?

     [

    Your

     Name

    ]

     is

     a

     creative

     and

     innovative

     thinker

     with

     a

     passion

     for

     [

    insert

     something

     that

     relates

     to

     your

     background

     or

     experience

     here

    ].

     I

     am

     always

     looking

     for

     new

     ideas

     and

     have

     a

     knack

     for

     coming

     up

     with

     unexpected

     solutions

    .

     I

    'm

     a

     team

     player

     and

     enjoy

     collaborating

     with

     others

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Light

    ,

     a

     historic

     and

     cultural

     center

     with

     a

     rich

     history

     dating

     back

     to

     ancient

     times

     and

     being

     one

     of

     the

     most

     populous

     cities

     in

     the

     world

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     of

     France

     and

     one

     of

     the

     most

     iconic

     and

     popular

     cities

     in

     the

     world

    .

     It

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

     and

     is

     considered

     one

     of

     the

     most

     beautiful

     and

     liv

    able

     cities

     in

     the

     world

    .

     Paris

     is

     known

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

     the

     Lou

    vre

     Museum

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     The

     city

     is

     also

     famous

     for

     its

     cuisine

    ,

     fashion

    ,

     and

     arts

     scene

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     range

     of

     technologies

     and

     applications

     that

     are

     currently

     in

     development

     or

     in

     various

     stages

     of

     development

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     Aug

    mented

     AI

    :

     Aug

    mented

     AI

     is

     a

     new

     AI

     approach

     that

     combines

     traditional

     machine

     learning

     algorithms

     with

     human

    -like

     understanding

     of

     natural

     language

     and

     context

    .

     This

     means

     that

     AI

     can

     learn

     from

     human

    -like

     behavior

     and

     can

     respond

     to

     human

    -like

     prompts

    .

     This

     technology

     could

     be

     used

     for

     things

     like

     virtual

     assistants

    ,

     chat

    bots

    ,

     and

     language

     translation

    .
    


    2

    .

     Neu

    rom

    orphic

     Computing

    :

     Neu

    rom

    orphic

     computing

     is

     a

     type

     of

     artificial

     intelligence

     that

     uses

     biological

     and

     neural

    -like

     mechanisms

     to

     process

     information

    .

     This

     technology

     could

     allow

    



```python
llm.shutdown()
```
