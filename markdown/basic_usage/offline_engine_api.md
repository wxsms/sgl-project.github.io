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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.00it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.00it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:30,  4.74s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.37it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.10it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  4.10it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:05<00:04,  8.71it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:05<00:04,  8.71it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:05<00:04,  8.71it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:05<00:04,  8.71it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:05<00:04,  8.71it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:05<00:04,  8.71it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:05<00:04,  8.71it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:05<00:04,  8.71it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:05<00:04,  8.71it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:05<00:04,  8.71it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 14.39it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 14.39it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 14.39it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 14.39it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 14.39it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 14.39it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 14.39it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 14.39it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:05<00:01, 14.39it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:05<00:01, 14.39it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:05<00:01, 14.39it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 22.05it/s]

    Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 22.05it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.81it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.81it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.81it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.81it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.81it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.81it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.81it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.81it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.81it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:05<00:00, 29.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.45 GB):   2%|▏         | 1/58 [00:00<00:06,  9.21it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   2%|▏         | 1/58 [00:00<00:06,  9.21it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:05,  9.46it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:05,  9.46it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.42 GB):   3%|▎         | 2/58 [00:00<00:05,  9.46it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.42 GB):   7%|▋         | 4/58 [00:00<00:05, 10.79it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.42 GB):   7%|▋         | 4/58 [00:00<00:05, 10.79it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):   7%|▋         | 4/58 [00:00<00:05, 10.79it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.41 GB):  10%|█         | 6/58 [00:00<00:04, 12.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.40 GB):  10%|█         | 6/58 [00:00<00:04, 12.99it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.40 GB):  10%|█         | 6/58 [00:00<00:04, 12.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  10%|█         | 6/58 [00:00<00:04, 12.99it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.40 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.39 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.11it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.38 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.37 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.98it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  22%|██▏       | 13/58 [00:00<00:02, 21.98it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.36 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  29%|██▉       | 17/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.73it/s]Capturing num tokens (num_tokens=960 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.73it/s] Capturing num tokens (num_tokens=896 avail_mem=74.35 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.73it/s]Capturing num tokens (num_tokens=832 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:01<00:01, 29.73it/s]Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  36%|███▌      | 21/58 [00:01<00:01, 29.73it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.18it/s]Capturing num tokens (num_tokens=704 avail_mem=74.34 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.18it/s]Capturing num tokens (num_tokens=640 avail_mem=74.33 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.18it/s]Capturing num tokens (num_tokens=576 avail_mem=74.33 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.18it/s]Capturing num tokens (num_tokens=512 avail_mem=74.32 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.18it/s]Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  43%|████▎     | 25/58 [00:01<00:01, 32.18it/s]Capturing num tokens (num_tokens=480 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=448 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=416 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=384 avail_mem=74.33 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=352 avail_mem=74.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.79it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  52%|█████▏    | 30/58 [00:01<00:00, 35.79it/s]Capturing num tokens (num_tokens=320 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 37.99it/s]Capturing num tokens (num_tokens=288 avail_mem=74.32 GB):  60%|██████    | 35/58 [00:01<00:00, 37.99it/s]Capturing num tokens (num_tokens=256 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 37.99it/s]Capturing num tokens (num_tokens=240 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 37.99it/s]Capturing num tokens (num_tokens=224 avail_mem=74.31 GB):  60%|██████    | 35/58 [00:01<00:00, 37.99it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  60%|██████    | 35/58 [00:01<00:00, 37.99it/s]Capturing num tokens (num_tokens=208 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.99it/s]Capturing num tokens (num_tokens=192 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.99it/s]Capturing num tokens (num_tokens=176 avail_mem=74.30 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.99it/s]Capturing num tokens (num_tokens=160 avail_mem=74.22 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.99it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.26 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.99it/s]Capturing num tokens (num_tokens=128 avail_mem=74.26 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.99it/s]Capturing num tokens (num_tokens=128 avail_mem=74.26 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=112 avail_mem=74.01 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=96 avail_mem=74.03 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.97it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.97it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.23 GB):  78%|███████▊  | 45/58 [00:01<00:00, 33.97it/s]Capturing num tokens (num_tokens=64 avail_mem=74.23 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.39it/s]Capturing num tokens (num_tokens=48 avail_mem=74.22 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.39it/s]Capturing num tokens (num_tokens=32 avail_mem=74.22 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.39it/s]Capturing num tokens (num_tokens=28 avail_mem=74.21 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.39it/s]Capturing num tokens (num_tokens=24 avail_mem=74.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 30.39it/s]Capturing num tokens (num_tokens=24 avail_mem=74.20 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.31it/s]Capturing num tokens (num_tokens=20 avail_mem=74.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.31it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.31it/s]Capturing num tokens (num_tokens=12 avail_mem=74.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 29.31it/s]Capturing num tokens (num_tokens=8 avail_mem=74.18 GB):  91%|█████████▏| 53/58 [00:02<00:00, 29.31it/s] Capturing num tokens (num_tokens=8 avail_mem=74.18 GB):  98%|█████████▊| 57/58 [00:02<00:00, 29.95it/s]Capturing num tokens (num_tokens=4 avail_mem=74.17 GB):  98%|█████████▊| 57/58 [00:02<00:00, 29.95it/s]Capturing num tokens (num_tokens=4 avail_mem=74.17 GB): 100%|██████████| 58/58 [00:02<00:00, 27.90it/s]


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
    Generated text:  Emily. I am a student of the College of Arts and Sciences at the University of Alabama at Birmingham, and I am a member of the Birmingham Center for Women. I recently returned to Birmingham to take part in the 2022 year of Black History Month. I have chosen to explore a few of the ways that Black women have fought for equality in the United States. I want to highlight a woman named Mary McLeod Bethune, who was a member of the Indian civil rights movement that united African Americans and Indian Americans. Mary McLeod Bethune was born in 1876 in Indianola, Alabama, and was
    ===============================
    Prompt: The president of the United States is
    Generated text:  supposed to speak about the economy at least once a year. If the president is still alive in 2050, then the president will have been dead for 30 years, when that year comes around. So, the economy will not be the main topic of the president's speech in 2050.
    
    Which of the following, if true, would most effectively refute the argument that the economy will not be the main topic of the president's speech in 2050?
    A: The president is likely to speak about other issues that are not focused on the economy.
    B: The president is likely to have
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. France has several cities that are important for various types of transport. The National Railways, which are a major transport operator, use a system of dedicated tracks to transport goods. This system is known as the National Railways' Linéaires, which are divided into different sections. The National Railways' Linéaires operate in France and are also used by the French Army. 
    
    In Paris, there are several stations that are important for rail transport. The National Railways use these stations to transport goods and passengers. The National Railways' Linéaires have a unique feature: they use a double track system. This means
    ===============================
    Prompt: The future of AI is
    Generated text:  set to be shaped by the benefits of regenerative medicine and the clinical uptake of these treatments. While those benefits will be realized in the next decade, a number of practical challenges must be addressed to fully realize their potential in the industry.
    Regenerative medicine’s success has been primarily due to the advances in stem cell technology and the proliferation of the field of regenerative medicine. Regenerative medicine is a growing field, but it has a limited impact on the industry. This is due to a number of practical challenges that can be addressed to realize the full potential of regenerative medicine.
    There are two main challenges that will have to be addressed if the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your job or experience here]. I enjoy [insert a brief description of your hobbies or interests here]. What's your favorite hobby or activity? I'm always looking for new experiences and learning new things, so I enjoy [insert a brief description of your hobby or activity here]. What's your favorite book or movie? I love [insert a brief description of your favorite book or movie here]. I'm always on
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville de Paris" and the "City of Light". It is the largest city in France and the second-largest city in the European Union. Paris is home to many iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also known for its rich cultural heritage, including the Notre-Dame Cathedral, the Louvre Museum, and the Palace of Versailles. Paris is a bustling city with a diverse population and is a major tourist destination. It is also known for its cuisine, fashion, and art scene. The city is home to many international organizations
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks with increasing accuracy and efficiency. Some potential future trends in AI include:
    
    1. Increased integration with human intelligence: AI systems are likely to become more integrated with human intelligence, allowing them to learn from and adapt to the behavior and preferences of humans. This could lead to more personalized and effective AI systems that can better understand and respond to the needs of their users.
    
    2. Greater use of AI in healthcare:
    


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
    Generated text:  [Your Name] and I'm here to meet you. How can I assist you today? I'm here to provide you with helpful information and answer any questions you may have. Please feel free to ask me anything and I'll do my best to answer you. How can I assist you today? I'm here to provide you with helpful information and answer any questions you may have. Please feel free to ask me anything and I'll do my best to answer you. How can I assist you today? I'm here to provide you with helpful information and answer any questions you may have. Please feel free to ask me anything and I'll
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the city of love and of learning. It is a sprawling city with a rich history, a vibrant cultural scene, and a cosmopolitan population of over 10 million people. Paris is a beloved city that is home to many iconic landmarks and attractions, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and many more. It is also the birthplace of many of France's most famous artists, writers, and musicians. Paris has a rich cultural heritage and continues to be a major hub of business, education, and entertainment in the world. Its iconic landmarks and iconic French culture make it
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to continue to grow and evolve, driven by a variety of trends and technologies that are shaping the field in new and exciting ways. Here are some possible future trends in AI:
    
    1. Increased focus on ethical and responsible AI: As more people become aware of the potential risks and biases in AI systems, there is growing pressure to design and build AI that is more ethically aligned with the values of human beings. This includes designing AI systems that are transparent, accountable, and accountable, and creating a culture of ethical AI that values the interests of all parties involved.
    
    2. Integration of AI into new industries and applications: As AI continues to


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

    Occup

    ation

    ].

     I

     have

     always

     been

     fascinated

     by

     the

     beauty

     of

     the

     natural

     world

     and

     have

     spent

     my

     entire

     life

     exploring

     and

     documenting

     it

    .

     I

     am

     a

     [

    Number

    ]

     of

     the

     [

    World

     Wildlife

     Fund

    ],

     working

     to

     protect

     the

     environment

     and

     promote

     sustainable

     living

    .

     I

     believe

     that

     everyone

     has

     the

     right

     to

     live

     in

     harmony

     with

     nature

     and

     that

     the

     earth

     is

     our

     home

    .

     I

     have

     a

     passion

     for

     learning

     and

     always

     aim

     to

     educate

     others

     about

     the

     importance

     of

     conservation

    .

     How

     can

     I

     help

     support

     the

     [

    World

     Wildlife

     Fund

    ]

     and

     what

     can

     I

     do

     to

     get

     involved

     in

     their

     work

    ?

     Hello

    ,

     my

     name

     is

     [

    Name

    ],

     and

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     statement

     is

    :

     The

     capital

     of

     France

     is

     Paris

    .

     
    


    Here

    's

     a

     step

    -by

    -step

     justification

     for

     this

     answer

    :
    


    1

    .

     Identify

     the

     capital

     of

     France

    :

     The

     capital

     of

     France

     is

     Paris

    .


    2

    .

     Form

    ulate

     the

     statement

    :

     The

     statement

     should

     be

     a

     concise

     factual

     statement

     that

     identifies

     the

     capital

     of

     France

     and

     includes

     its

     name

    .


    3

    .

     Provide

     context

    :

     The

     statement

     should

     include

     the

     relevant

     context

    ,

     such

     as

     the

     importance

     of

     Paris

     as

     a

     major

     French

     city

    .
    


    The

     statement

     "

    The

     capital

     of

     France

     is

     Paris

    "

     is

     a

     clear

     and

     concise

     factual

     statement

     about

     the

     capital

     city

     of

     France

    .

     This

     answer

     provides

     a

     straightforward

     and

     accurate

     description

     of

     the

     capital

    's

     location

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     undoubtedly

     bright

    ,

     with

     a

     lot

     of

     different

     trends

     expected

     to

     emerge

     in

     the

     coming

     years

    .

     Some

     of

     the

     most

     prominent

     trends

     in

     AI

     include

    :
    


    1

    .

     Increased

     integration

     with

     human

     emotion

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     is

     likely

     that

     it

     will

     be

     able

     to

     interpret

     and

     understand

     human

     emotions

    ,

     emotions

    ,

     and

     even

     be

     able

     to

     interpret

     and

     understand

     human

     emotions

    .
    


    2

    .

     Increased

     reliance

     on

     AI

     for

     decision

    -making

    :

     With

     AI

     becoming

     more

     capable

     of

     making

     decisions

    ,

     it

     is

     likely

     that

     more

     and

     more

     organizations

     will

     be

     using

     AI

     for

     decision

    -making

    ,

     reducing

     the

     need

     for

     human

     decision

    -makers

    .
    


    3

    .

     AI

     becoming

     more

     autonomous

    :

     As

     AI

     technology

     continues

     to

     improve

    ,

     it

     is

    



```python
llm.shutdown()
```
