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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.71it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.70it/s]


    2026-04-29 21:01:12,385 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 21:01:12] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:38,  4.89s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:51,  1.04it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:51,  1.04it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:05<00:51,  1.04it/s]

    Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:05<00:51,  1.04it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:05<00:23,  2.14it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:05<00:23,  2.14it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:05<00:23,  2.14it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:05<00:23,  2.14it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:05<00:23,  2.14it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:11,  4.05it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:11,  4.05it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:11,  4.05it/s]

    Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:11,  4.05it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:11,  4.05it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:11,  4.05it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=1792):  28%|██▊       | 16/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=1536):  28%|██▊       | 16/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=1280):  28%|██▊       | 16/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=1024):  28%|██▊       | 16/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=960):  28%|██▊       | 16/58 [00:05<00:05,  7.13it/s] Compiling num tokens (num_tokens=896):  28%|██▊       | 16/58 [00:05<00:05,  7.13it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 12.52it/s]

    Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=448):  40%|███▉      | 23/58 [00:05<00:02, 12.52it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 19.77it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 28.82it/s]

    Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 28.82it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 28.82it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 36.67it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 36.67it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 36.67it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 36.67it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 36.67it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 36.67it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 36.67it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 36.67it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 36.67it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 43.67it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 43.67it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 43.67it/s]

    Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.75it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.18 GB):   3%|▎         | 2/58 [00:00<00:05, 11.13it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.36 GB):   3%|▎         | 2/58 [00:00<00:05, 11.13it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.35 GB):   3%|▎         | 2/58 [00:00<00:05, 11.13it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.35 GB):   7%|▋         | 4/58 [00:00<00:04, 13.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.35 GB):   7%|▋         | 4/58 [00:00<00:04, 13.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.34 GB):   7%|▋         | 4/58 [00:00<00:04, 13.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.34 GB):  10%|█         | 6/58 [00:00<00:03, 15.53it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.33 GB):  10%|█         | 6/58 [00:00<00:03, 15.53it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.21 GB):  10%|█         | 6/58 [00:00<00:03, 15.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.22 GB):  10%|█         | 6/58 [00:00<00:03, 15.53it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.22 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.24 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.24 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.24 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.66it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.24 GB):  21%|██        | 12/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.24 GB):  21%|██        | 12/58 [00:00<00:02, 21.62it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.24 GB):  21%|██        | 12/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.24 GB):  21%|██        | 12/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.24 GB):  21%|██        | 12/58 [00:00<00:02, 21.62it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.24 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.35it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.25 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.35it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.26 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.35it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.25 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.35it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.25 GB):  28%|██▊       | 16/58 [00:00<00:01, 25.35it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=74.25 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.23 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.80it/s]Capturing num tokens (num_tokens=960 avail_mem=74.24 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.80it/s] Capturing num tokens (num_tokens=896 avail_mem=74.23 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.80it/s]Capturing num tokens (num_tokens=832 avail_mem=74.22 GB):  34%|███▍      | 20/58 [00:00<00:01, 28.80it/s]Capturing num tokens (num_tokens=832 avail_mem=74.22 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=768 avail_mem=74.23 GB):  41%|████▏     | 24/58 [00:00<00:01, 31.94it/s]Capturing num tokens (num_tokens=704 avail_mem=74.22 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.94it/s]Capturing num tokens (num_tokens=640 avail_mem=74.22 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.94it/s]Capturing num tokens (num_tokens=576 avail_mem=74.21 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.94it/s]Capturing num tokens (num_tokens=512 avail_mem=74.19 GB):  41%|████▏     | 24/58 [00:01<00:01, 31.94it/s]

    Capturing num tokens (num_tokens=512 avail_mem=74.19 GB):  50%|█████     | 29/58 [00:01<00:00, 35.06it/s]Capturing num tokens (num_tokens=480 avail_mem=74.21 GB):  50%|█████     | 29/58 [00:01<00:00, 35.06it/s]Capturing num tokens (num_tokens=448 avail_mem=74.20 GB):  50%|█████     | 29/58 [00:01<00:00, 35.06it/s]Capturing num tokens (num_tokens=416 avail_mem=74.19 GB):  50%|█████     | 29/58 [00:01<00:00, 35.06it/s]Capturing num tokens (num_tokens=384 avail_mem=74.19 GB):  50%|█████     | 29/58 [00:01<00:00, 35.06it/s]Capturing num tokens (num_tokens=352 avail_mem=74.18 GB):  50%|█████     | 29/58 [00:01<00:00, 35.06it/s]Capturing num tokens (num_tokens=352 avail_mem=74.18 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=320 avail_mem=74.17 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=288 avail_mem=74.17 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=256 avail_mem=74.14 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.53it/s]

    Capturing num tokens (num_tokens=240 avail_mem=74.14 GB):  59%|█████▊    | 34/58 [00:01<00:00, 36.53it/s]Capturing num tokens (num_tokens=240 avail_mem=74.14 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=224 avail_mem=74.13 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=208 avail_mem=74.12 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=192 avail_mem=74.14 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=176 avail_mem=74.13 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=160 avail_mem=74.13 GB):  66%|██████▌   | 38/58 [00:01<00:00, 36.81it/s]Capturing num tokens (num_tokens=160 avail_mem=74.13 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=144 avail_mem=74.12 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=128 avail_mem=74.11 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=112 avail_mem=74.11 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.64it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.10 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.64it/s] Capturing num tokens (num_tokens=80 avail_mem=74.09 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.64it/s]Capturing num tokens (num_tokens=80 avail_mem=74.09 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=64 avail_mem=74.09 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=48 avail_mem=74.08 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=32 avail_mem=74.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=28 avail_mem=74.07 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  83%|████████▎ | 48/58 [00:01<00:00, 39.49it/s]Capturing num tokens (num_tokens=24 avail_mem=74.06 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.04it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.04it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.04it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.05 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.04it/s]Capturing num tokens (num_tokens=8 avail_mem=74.05 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.04it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  91%|█████████▏| 53/58 [00:01<00:00, 40.04it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 42.42it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 32.29it/s]


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
    Generated text:  Jerald Martinez. My name is Tom, and I am a current student of the College of Public Health at the University of Florida. I am a physical therapist and associate professor at the College of Public Health, where I teach health physical therapy and the biomechanics of motor skills. I am also a medical physical therapist and associate professor at the University of Georgia. My research is in the area of biomechanics, and I have published research on the biomechanics of balance and gait in high school students. I am a 2023 National Physical Therapy Association Education Fellow and a member of the American Physical Therapy Association.
    ===============================
    Prompt: The president of the United States is
    Generated text:  3 feet tall. If a politician is 2 feet shorter than the president, and the other politician is 3 feet taller than the president, how tall is the other politician in inches? First, let's determine the height of the politician who is 2 feet shorter than the president. The president is 3 feet tall, so 2 feet is 2 times 1 foot, which is 2 feet. Therefore, the height of the other politician is:
    
    \[ 3 \text{ feet} - 2 \text{ feet} = 1 \text{ foot} \]
    
    Next, let's determine the height
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It was named after the ancient Greek city of Athens, on the bay of the same name. In the 19th century, Paris was a national capital. In 1792, the name of the city was changed to Paris. The city has a population of 2.13 million in 2014. Paris has been used as the capital city of France since 812 BC. 
    
    A survey was conducted in 2015 to find the population of Paris. The results were as follows:
    - In 2015, there were 2.35 million
    ===============================
    Prompt: The future of AI is
    Generated text:  here, but the question is what will make it true? As the world's most influential AI company, we are tasked with unlocking the full potential of AI and working with our partners to create solutions that can be used by all of society to their advantage. We are exploring the future of AI with a focus on advancements in areas such as speech recognition, natural language processing, computer vision, robotics, and machine learning.
    In this article, we explore the current state of AI and discuss some of the trends and challenges that we are currently facing. We also consider the future of AI and what it will mean for society as a whole.
    The Future


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


    Generated text:  [Name] and I am a [occupation] who has been [number of years] in the industry. I am passionate about [reason for passion], and I am always looking for ways to [action or goal]. I am [age] years old, and I have a [number of hobbies or interests] that I enjoy. I am [gender] and I am [number of children] years old. I am [occupation] and I have been [number of years] in the industry. I am passionate about [reason for passion], and I am always looking for ways to [action or goal]. I am [age]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city that is known for its iconic Eiffel Tower, Notre-Dame Cathedral, and the annual Eiffel Tower Festival. It is also the birthplace of French writer Victor Hugo and the home of the Louvre Museum. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also home to the French Parliament, the French Parliament building, and the Eiffel Tower. The city is known for its diverse cuisine, fashion, and art scene. Paris is a city that is a true reflection of French culture and history. It is a city that is a must-visit
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation and robotics: As AI technology continues to advance, we can expect to see more automation and robotics in various industries. This could lead to increased efficiency, lower costs, and better quality of life for many people.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even more sophisticated applications in this area.
    
    3. AI-powered education: AI is already
    


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
    Generated text:  [Name] and I am a [occupation] who is passionate about [career goal]. I am confident in my abilities and am eager to make a positive impact in my field, whether it be through my work, my relationships, or my community. I am always up for a challenge and I am eager to learn new things and grow as a person. I am a team player and thrive in collaborative environments. I am committed to using my skills and knowledge to make a difference in the world and I am determined to achieve my career goals. Thank you. I hope you find this introduction interesting. Let me know if you would like me to
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    That is the largest city in France, and it's the oldest in the world. The city is located in the central part of France, near the Mediterranean Sea, on the western bank of the Seine River. It has a population of over 1 million people and is known for its rich history and diverse cultural scene. Paris is a vibrant and cosmopolitan city with a famous array of landmarks and attractions, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It's also home to many famous museums, theaters, and restaurants, and is known for its delicious cuisine, including croissants,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  exciting and rapidly evolving. Some possible trends in AI include:
    
    1. Increased complexity and sophistication: As AI systems become more complex and sophisticated, they will become even more capable of performing tasks that were previously considered too challenging or dangerous for humans.
    
    2. Integration with other technologies: AI systems will become increasingly integrated with other technologies, such as sensors, robots, and virtual assistants, creating a more complete and interconnected system.
    
    3. Autonomous decision-making: As AI systems become more advanced, they will be able to make decisions that are more independent and responsible. This could lead to a greater emphasis on AI systems that can make autonomous decisions and take responsibility


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

     ____

    _.

     I

    'm

     a

    /an

     [

    insert

     profession

     or

     occupation

     here

    ,

     such

     as

     "

    teacher

    ,"

     "

    engine

    er

    ,"

     "

    writer

    ,"

     etc

    .

     ]

     and

     I

    've

     been

     working

     in

     the

     field

     of

     [

    insert

     field

     here

    ,

     such

     as

     "

    education

    ,"

     "

    technology

    ,"

     "

    environment

    al

     science

    ,"

     etc

    .

     ].

     I

    'm

     passionate

     about

     [

    insert

     something

     that

     reflects

     your

     interests

     or

     hobbies

     here

    ,

     such

     as

     "

    reading

    ,"

     "

    travel

    ing

    ,"

     "

    science

    ,"

     "

    art

    ,"

     etc

    .

     ].

     My

     work

     ethic

     is

     [

    insert

     a

     trait

     or

     quality

     here

    ,

     such

     as

     "

    professional

    ,"

     "

    ded

    icated

    ,"

     "

    amb

    itious

    ,"

     "

    hard

    working

    ,"

     "

    gr

    acious

    ,"

     "

    ded

    icated

    ,"

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     Europe

     and

     the

     largest

     metropolitan

     area

     in

     the

     world

    .

     It

     is

     the

     seat

     of

     government

     for

     France

     and

     serves

     as

     a

     major

     commercial

    ,

     cultural

    ,

     and

     intellectual

     hub

    .

     Paris

     has

     a

     rich

     history

     and

     cultural

     heritage

    ,

     including

     ancient

     Roman

     ruins

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     also

     has

     a

     diverse

     population

    ,

     including

     French

    ,

     French

    -speaking

     immigrants

    ,

     and

     various

     ethnic

    ities

    .

     Paris

     is

     also

     home

     to

     numerous

     cultural

     and

     artistic

     institutions

    ,

     including

     the

     Op

    éra

    ,

     the

     Lou

    vre

    ,

     and

     the

     Mus

    ée

     d

    '

    Or

    say

    .

     It

     is

     known

     for

     its

     scenic

     beauty

    ,

     including

     the

     Se

    ine

     River

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     see

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

     autonomous

     systems

    .

     These

     developments

     could

     have

     a

     wide

     range

     of

     implications

     for

     society

    ,

     from

     job

     displacement

     to

     increased

     privacy

     and

     security

     concerns

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

     use

     of

     AI

     in

     healthcare

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     may

     be

     used

     to

     diagnose

     and

     treat

     diseases

     more

     accurately

     and

     quickly

     than

     human

     doctors

    .

     This

     could

     have

     a

     significant

     impact

     on

     healthcare

     outcomes

    ,

     and

     could

     potentially

     lead

     to

     new

     medical

     breakthrough

    s

    .
    


    2

    .

     AI

     in

     education

    :

     AI

    -powered

     educational

     tools

     and

     systems

     could

     provide

     personalized

     learning

     experiences

    ,

     improve

     student

     engagement

     and

     achievement

    ,

     and

     help

    



```python
llm.shutdown()
```
