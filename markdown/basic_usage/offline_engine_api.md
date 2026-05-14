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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.83it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.82it/s]


    2026-05-14 02:30:43,628 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-14 02:30:43] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:08,  4.35s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:04,  1.18s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:04,  1.18s/it]

    Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:04,  1.18s/it]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:04,  1.18s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:24,  2.08it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:24,  2.08it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:24,  2.08it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:24,  2.08it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:24,  2.08it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:04<00:24,  2.08it/s]Compiling num tokens (num_tokens=3328):  10%|█         | 6/58 [00:04<00:24,  2.08it/s]Compiling num tokens (num_tokens=3072):  10%|█         | 6/58 [00:04<00:24,  2.08it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:04<00:07,  5.94it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:04<00:07,  5.94it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:04<00:07,  5.94it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:04<00:07,  5.94it/s]Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:04<00:07,  5.94it/s]

    Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:04<00:07,  5.94it/s]Compiling num tokens (num_tokens=1536):  22%|██▏       | 13/58 [00:04<00:07,  5.94it/s]Compiling num tokens (num_tokens=1280):  22%|██▏       | 13/58 [00:04<00:07,  5.94it/s]Compiling num tokens (num_tokens=1024):  22%|██▏       | 13/58 [00:04<00:07,  5.94it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03, 11.48it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03, 11.48it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03, 11.48it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03, 11.48it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03, 11.48it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03, 11.48it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03, 11.48it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03, 11.48it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03, 11.48it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03, 11.48it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 18.91it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 18.91it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 18.91it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 18.91it/s]

    Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 18.91it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 18.91it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 18.91it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 18.91it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 18.91it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 18.91it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 18.91it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 28.42it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 28.42it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 35.72it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 35.72it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 35.72it/s]

    Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 35.72it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 35.72it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 35.72it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 35.72it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 35.72it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 35.72it/s]Compiling num tokens (num_tokens=12):  97%|█████████▋| 56/58 [00:05<00:00, 42.05it/s]Compiling num tokens (num_tokens=8):  97%|█████████▋| 56/58 [00:05<00:00, 42.05it/s] Compiling num tokens (num_tokens=4):  97%|█████████▋| 56/58 [00:05<00:00, 42.05it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.97it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=37.82 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=37.82 GB):   2%|▏         | 1/58 [00:00<00:06,  9.06it/s]Capturing num tokens (num_tokens=7680 avail_mem=37.79 GB):   2%|▏         | 1/58 [00:00<00:06,  9.06it/s]Capturing num tokens (num_tokens=7168 avail_mem=37.78 GB):   2%|▏         | 1/58 [00:00<00:06,  9.06it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=37.78 GB):   5%|▌         | 3/58 [00:00<00:03, 15.21it/s]Capturing num tokens (num_tokens=6656 avail_mem=37.78 GB):   5%|▌         | 3/58 [00:00<00:03, 15.21it/s]Capturing num tokens (num_tokens=6144 avail_mem=37.78 GB):   5%|▌         | 3/58 [00:00<00:03, 15.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=37.78 GB):   5%|▌         | 3/58 [00:00<00:03, 15.21it/s]Capturing num tokens (num_tokens=5632 avail_mem=37.78 GB):  10%|█         | 6/58 [00:00<00:02, 20.08it/s]Capturing num tokens (num_tokens=5120 avail_mem=37.77 GB):  10%|█         | 6/58 [00:00<00:02, 20.08it/s]Capturing num tokens (num_tokens=4608 avail_mem=37.76 GB):  10%|█         | 6/58 [00:00<00:02, 20.08it/s]Capturing num tokens (num_tokens=4096 avail_mem=37.76 GB):  10%|█         | 6/58 [00:00<00:02, 20.08it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=37.76 GB):  10%|█         | 6/58 [00:00<00:02, 20.08it/s]Capturing num tokens (num_tokens=3840 avail_mem=37.76 GB):  17%|█▋        | 10/58 [00:00<00:01, 25.51it/s]Capturing num tokens (num_tokens=3584 avail_mem=37.75 GB):  17%|█▋        | 10/58 [00:00<00:01, 25.51it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.35 GB):  17%|█▋        | 10/58 [00:00<00:01, 25.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.35 GB):  17%|█▋        | 10/58 [00:00<00:01, 25.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.35 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.35 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.82it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=59.34 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.34 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.33 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.33 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.82it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.33 GB):  31%|███       | 18/58 [00:00<00:01, 28.50it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.33 GB):  31%|███       | 18/58 [00:00<00:01, 28.50it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.33 GB):  31%|███       | 18/58 [00:00<00:01, 28.50it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.31 GB):  31%|███       | 18/58 [00:00<00:01, 28.50it/s]Capturing num tokens (num_tokens=960 avail_mem=59.32 GB):  31%|███       | 18/58 [00:00<00:01, 28.50it/s] Capturing num tokens (num_tokens=896 avail_mem=59.32 GB):  31%|███       | 18/58 [00:00<00:01, 28.50it/s]

    Capturing num tokens (num_tokens=896 avail_mem=59.32 GB):  40%|███▉      | 23/58 [00:00<00:01, 33.46it/s]Capturing num tokens (num_tokens=832 avail_mem=59.32 GB):  40%|███▉      | 23/58 [00:00<00:01, 33.46it/s]Capturing num tokens (num_tokens=768 avail_mem=59.31 GB):  40%|███▉      | 23/58 [00:00<00:01, 33.46it/s]Capturing num tokens (num_tokens=704 avail_mem=59.31 GB):  40%|███▉      | 23/58 [00:00<00:01, 33.46it/s]Capturing num tokens (num_tokens=640 avail_mem=59.31 GB):  40%|███▉      | 23/58 [00:00<00:01, 33.46it/s]Capturing num tokens (num_tokens=640 avail_mem=59.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.44it/s]Capturing num tokens (num_tokens=576 avail_mem=59.31 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.44it/s]Capturing num tokens (num_tokens=512 avail_mem=59.29 GB):  47%|████▋     | 27/58 [00:00<00:00, 34.44it/s]Capturing num tokens (num_tokens=480 avail_mem=59.31 GB):  47%|████▋     | 27/58 [00:01<00:00, 34.44it/s]Capturing num tokens (num_tokens=448 avail_mem=59.30 GB):  47%|████▋     | 27/58 [00:01<00:00, 34.44it/s]Capturing num tokens (num_tokens=416 avail_mem=59.30 GB):  47%|████▋     | 27/58 [00:01<00:00, 34.44it/s]

    Capturing num tokens (num_tokens=416 avail_mem=59.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=384 avail_mem=59.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=352 avail_mem=59.29 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=320 avail_mem=59.29 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=288 avail_mem=59.29 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=256 avail_mem=59.28 GB):  55%|█████▌    | 32/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=256 avail_mem=59.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.10it/s]Capturing num tokens (num_tokens=240 avail_mem=59.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.10it/s]

    Capturing num tokens (num_tokens=224 avail_mem=59.28 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.10it/s]Capturing num tokens (num_tokens=208 avail_mem=59.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.10it/s]Capturing num tokens (num_tokens=192 avail_mem=59.27 GB):  64%|██████▍   | 37/58 [00:01<00:00, 37.10it/s]

    Capturing num tokens (num_tokens=192 avail_mem=59.27 GB):  71%|███████   | 41/58 [00:01<00:00, 23.92it/s]Capturing num tokens (num_tokens=176 avail_mem=59.27 GB):  71%|███████   | 41/58 [00:01<00:00, 23.92it/s]Capturing num tokens (num_tokens=160 avail_mem=59.27 GB):  71%|███████   | 41/58 [00:01<00:00, 23.92it/s]Capturing num tokens (num_tokens=144 avail_mem=59.26 GB):  71%|███████   | 41/58 [00:01<00:00, 23.92it/s]

    Capturing num tokens (num_tokens=128 avail_mem=59.26 GB):  71%|███████   | 41/58 [00:01<00:00, 23.92it/s]Capturing num tokens (num_tokens=128 avail_mem=59.26 GB):  78%|███████▊  | 45/58 [00:01<00:00, 18.96it/s]Capturing num tokens (num_tokens=112 avail_mem=59.26 GB):  78%|███████▊  | 45/58 [00:01<00:00, 18.96it/s]Capturing num tokens (num_tokens=96 avail_mem=59.25 GB):  78%|███████▊  | 45/58 [00:01<00:00, 18.96it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=59.25 GB):  78%|███████▊  | 45/58 [00:02<00:00, 18.96it/s]Capturing num tokens (num_tokens=80 avail_mem=59.25 GB):  83%|████████▎ | 48/58 [00:02<00:00, 17.04it/s]Capturing num tokens (num_tokens=64 avail_mem=59.25 GB):  83%|████████▎ | 48/58 [00:02<00:00, 17.04it/s]Capturing num tokens (num_tokens=48 avail_mem=59.24 GB):  83%|████████▎ | 48/58 [00:02<00:00, 17.04it/s]

    Capturing num tokens (num_tokens=32 avail_mem=59.24 GB):  83%|████████▎ | 48/58 [00:02<00:00, 17.04it/s]Capturing num tokens (num_tokens=32 avail_mem=59.24 GB):  88%|████████▊ | 51/58 [00:02<00:00, 15.71it/s]Capturing num tokens (num_tokens=28 avail_mem=59.24 GB):  88%|████████▊ | 51/58 [00:02<00:00, 15.71it/s]Capturing num tokens (num_tokens=24 avail_mem=59.23 GB):  88%|████████▊ | 51/58 [00:02<00:00, 15.71it/s]

    Capturing num tokens (num_tokens=24 avail_mem=59.23 GB):  91%|█████████▏| 53/58 [00:02<00:00, 15.01it/s]Capturing num tokens (num_tokens=20 avail_mem=59.23 GB):  91%|█████████▏| 53/58 [00:02<00:00, 15.01it/s]Capturing num tokens (num_tokens=16 avail_mem=59.23 GB):  91%|█████████▏| 53/58 [00:02<00:00, 15.01it/s]Capturing num tokens (num_tokens=16 avail_mem=59.23 GB):  95%|█████████▍| 55/58 [00:02<00:00, 14.69it/s]Capturing num tokens (num_tokens=12 avail_mem=59.22 GB):  95%|█████████▍| 55/58 [00:02<00:00, 14.69it/s]

    Capturing num tokens (num_tokens=8 avail_mem=59.22 GB):  95%|█████████▍| 55/58 [00:02<00:00, 14.69it/s] Capturing num tokens (num_tokens=8 avail_mem=59.22 GB):  98%|█████████▊| 57/58 [00:02<00:00, 14.23it/s]Capturing num tokens (num_tokens=4 avail_mem=59.22 GB):  98%|█████████▊| 57/58 [00:02<00:00, 14.23it/s]Capturing num tokens (num_tokens=4 avail_mem=59.22 GB): 100%|██████████| 58/58 [00:02<00:00, 20.30it/s]


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
    Generated text:  Yara Mehta, I am a graduate student in math. I have been asked to present a paper in a conference, but I am not sure how to arrange it.
    
    It is a one-page paper with five parts. Each part has a different title and subheadings. Here is the abstract:
    
    Part I. Motivation and the Problem
    Motivation and the Problem
    Motivation and the Problem
    
    Part II. Mathematical Formulation
    Mathematical Formulation
    Mathematical Formulation
    
    Part III. Real World Application
    Real World Application
    Real World Application
    
    Part IV. Computational Complexity
    Computational Complexity
    Computational
    ===============================
    Prompt: The president of the United States is
    Generated text:  a type of ____
    A. Political party
    B. Government agency
    C. Political party and government agency
    D. Government department
    Answer: C
    
    Which of the following does NOT conform to the rules of grammar?
    A. Amy is going to see the movie on Sunday.
    B. Amy will go to see the movie on Sunday.
    C. Amy will see the movie on Sunday.
    D. Amy is going to see the movie on Sunday.
    Answer: C
    
    The "People's Republic of China Safety Production Law" stipulates that when the construction unit and the safety production supervision and administration department carry out safety production inspections, they
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest city of France with an area of over 78 square kilometres (30 sq mi). It is located at the western end of the Île de France, at the north-eastern tip of the French Riviera, in the central region of the Ile-de-France metropolitan area. The city is situated at the mouth of the Seine, across from the river’s source, and it is the closest city to the mouth of the Seine (in the direction of Lyon, the city on which the Seine is the main artery of French rivers). With its location on the Seine, Paris
    ===============================
    Prompt: The future of AI is
    Generated text:  not about building robotic servants; it’s about developing highly intelligent, artificial superhumans. In the film Forbidden Planet, writer-director Steven Spielberg imagines the release of what?
    
    A) A superhuman being from a futuristic civilization
    B) A real person from a human society
    C) An artificial intelligence created by humans
    D) A robot from the stars
    
    D) A robot from the stars
    
    The film "Forrest Gump" by Robert Zemeckis features a character named Arnold (played by Robert De Niro) who is described as a "robotic cowboy" and a "robotic cowboy" who was


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [Age], [Gender], [Nationality], [Occupation], and I have [Number of Years] years of experience in [Field/Industry]. I'm a [Favorite Color], [Favorite Book], [Favorite Movie], and [Favorite Sport]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I'm always looking for new ways to challenge myself and expand my knowledge. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is known for its rich history, beautiful architecture, and vibrant culture. Paris is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. It is also a major center for business, finance, and tourism. Paris is a popular tourist destination and a cultural hub for France and the world. The city is home to many museums, theaters, and other cultural institutions, and is known for its annual festivals and events. Paris is a city of contrasts, with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is likely to become more prevalent in many industries, with automation becoming a more common feature of work. This could lead to job losses in some sectors, but also create new opportunities for people to work in areas that require automation.
    
    2. AI ethics and privacy: As AI becomes more prevalent, there will be a growing concern about its impact on society. This includes questions about how AI should be used and how it should be regulated. There will also be
    


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
    Generated text:  [Name], and I'm [Your Age]. I grew up in [City], [State]. I enjoy [activity], and I'm a [profession or hobby]. In my free time, I like to [optional activity, such as reading, painting, or playing music]. I've always been [character trait or characteristic], and I believe in [commitment to something, such as learning, charity, or environmental conservation]. If you're interested in learning more about me, I'd be happy to share some [personal interest, such as my interests in sports or music]. And if you want to know more about [character or hobby
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe and a popular tourist destination. Paris is renowned for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Despite its size, Paris is known for its livable quality, with its affordable living costs and extensive public transportation system. The city also hosts numerous cultural events and festivals throughout the year. Paris is a city that provides a unique blend of history, culture, and modernity for its residents and visitors alike. The capital of France is undoubtedly a must-visit destination
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a rapidly evolving field with many potential trends and developments. Some of the most significant trends include:
    
    1. AI automation: AI will continue to become more and more automated, reducing the need for manual intervention in many areas. This will include areas such as customer service, manufacturing, and data analysis.
    
    2. AI ethics and bias: As AI continues to gain more control over our lives, there will be an increasing emphasis on ethical guidelines and the responsible development of AI. This will involve looking at AI systems from multiple perspectives, including the impact on human rights, privacy, and fairness.
    
    3. AI for healthcare: AI is already being used to


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

    ]

     and

     I

    'm

     a

     [

    Your

     occupation

    ]

     who

     has

     a

     deep

    -se

    ated

     desire

     to

     learn

     and

     grow

    .

     I

     have

     a

     unique

     and

     humorous

     approach

     to

     problem

    -solving

    ,

     often

     finding

     unexpected

     solutions

     that

     benefit

     others

    .

     I

    'm

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     learn

     and

     improve

    .

     I

     enjoy

     sharing

     my

     knowledge

     and

     skills

     with

     others

    ,

     and

     I

    'm

     always

     eager

     to

     help

     them

     succeed

     in

     whatever

     they

     set

     out

     to

     do

    .

     I

    'm

     confident

     and

     hard

    working

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     make

     the

     world

     a

     better

     place

    .

     Thank

     you

     for

     having

     me

    .

     [

    Your

     Name

    ]

     [

    Occup

    ation

    ]

     [

    Your

     Profession

    ]

     Hello

    ,

     my

     name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

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

     
    


    I

     have

     a

     few

     more

     questions

     about

     Paris

     that

     I

     need

     to

     prepare

     for

     my

     interview

    :
    


    1

    .

     What

     are

     some

     of

     the

     iconic

     landmarks

     and

     buildings

     in

     Paris

    ,

     and

     how

     do

     they

     reflect

     the

     city

    's

     history

     and

     culture

    ?
    


    2

    .

     How

     has

     Paris

     been

     a

     hub

     of

     innovation

     and

     artistic

     expression

     throughout

     its

     history

    ,

     and

     what

     role

     has

     it

     played

     in

     shaping

     contemporary

     Paris

    ?
    


    3

    .

     Describe

     Paris

    's

     public

     transportation

     system

     and

     how

     it

     has

     evolved

     over

     the

     years

    ,

     including

     changes

     in

     pricing

    ,

     frequency

    ,

     and

     routes

    .
    


    4

    .

     What

     are

     some

     of

     the

     festivals

     and

     events

     that

     take

     place

     in

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     involve

     a

     wide

     range

     of

     developments

     that

     are

     both

     exciting

     and

     challenging

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

     Greater

     integration

     of

     AI

     into

     everyday

     life

    :

     As

     AI

     becomes

     more

     advanced

     and

     more

     widely

     available

    ,

     we

     can

     expect

     to

     see

     more

     AI

     technology

     integrated

     into

     our

     everyday

     lives,

     from

     self

    -driving

     cars

     to

     chat

    bots

     that

     can

     understand

     and

     respond

     to

     customer

     queries

    .
    


    2

    .

     Increased

     transparency

     and

     accountability

     of

     AI

     systems

    :

     AI

     systems

     are

     becoming

     more

     sophisticated

     and

     can

     operate

     on

     their

     own

     without

     human

     intervention

    ,

     leading

     to

     concerns

     about

     the

     ethical

     implications

     of

     these

     systems

    .

     As

     a

     result

    ,

     there

     may

     be

     a

     greater

     emphasis

     on

     increasing

     transparency

     and

     accountability

     in

     AI

    



```python
llm.shutdown()
```
