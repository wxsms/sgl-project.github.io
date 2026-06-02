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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.61it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.60it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:59,  4.19s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:59,  4.19s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:59,  4.19s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:01,  1.13s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:01,  1.13s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:01,  1.13s/it]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:01,  1.13s/it]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:23,  2.19it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:23,  2.19it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:23,  2.19it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:23,  2.19it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:04<00:23,  2.19it/s]Compiling num tokens (num_tokens=3584):  10%|█         | 6/58 [00:04<00:23,  2.19it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:09,  5.01it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:09,  5.01it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:09,  5.01it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:09,  5.01it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:09,  5.01it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:09,  5.01it/s]

    Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:09,  5.01it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:04,  9.30it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:04,  9.30it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:04,  9.30it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:04,  9.30it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:04,  9.30it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:04<00:04,  9.30it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:04<00:04,  9.30it/s]Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:04<00:04,  9.30it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:04<00:02, 15.29it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:04<00:02, 15.29it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:04<00:02, 15.29it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:04<00:02, 15.29it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:04<00:02, 15.29it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:04<00:02, 15.29it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:04<00:02, 15.29it/s]

    Compiling num tokens (num_tokens=448):  41%|████▏     | 24/58 [00:04<00:02, 15.29it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 22.10it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 22.10it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 22.10it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 22.10it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 22.10it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 22.10it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 22.10it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 22.10it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 22.10it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 30.62it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 30.62it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 30.62it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 30.62it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 30.62it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 30.62it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 30.62it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 30.62it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:05<00:00, 30.62it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 39.16it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 39.16it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 39.16it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 39.16it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 39.16it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 39.16it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 39.16it/s]Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 39.16it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 39.16it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 39.16it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 39.16it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 39.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 52.65it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.27it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=68.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.61 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=68.61 GB):   3%|▎         | 2/58 [00:00<00:05, 10.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=68.43 GB):   3%|▎         | 2/58 [00:00<00:05, 10.53it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=68.45 GB):   3%|▎         | 2/58 [00:00<00:05, 10.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=68.45 GB):   7%|▋         | 4/58 [00:00<00:04, 11.27it/s]Capturing num tokens (num_tokens=6144 avail_mem=68.61 GB):   7%|▋         | 4/58 [00:00<00:04, 11.27it/s]Capturing num tokens (num_tokens=5632 avail_mem=68.60 GB):   7%|▋         | 4/58 [00:00<00:04, 11.27it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=68.60 GB):  10%|█         | 6/58 [00:00<00:04, 12.69it/s]Capturing num tokens (num_tokens=5120 avail_mem=68.59 GB):  10%|█         | 6/58 [00:00<00:04, 12.69it/s]Capturing num tokens (num_tokens=4608 avail_mem=68.58 GB):  10%|█         | 6/58 [00:00<00:04, 12.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.58 GB):  10%|█         | 6/58 [00:00<00:04, 12.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=68.58 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=3840 avail_mem=68.57 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.82it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=68.57 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=3328 avail_mem=68.56 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.82it/s]Capturing num tokens (num_tokens=3328 avail_mem=68.56 GB):  21%|██        | 12/58 [00:00<00:02, 16.89it/s]Capturing num tokens (num_tokens=3072 avail_mem=68.55 GB):  21%|██        | 12/58 [00:00<00:02, 16.89it/s]Capturing num tokens (num_tokens=2816 avail_mem=68.55 GB):  21%|██        | 12/58 [00:00<00:02, 16.89it/s]Capturing num tokens (num_tokens=2560 avail_mem=68.54 GB):  21%|██        | 12/58 [00:00<00:02, 16.89it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=68.54 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=2304 avail_mem=68.53 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=2048 avail_mem=68.50 GB):  26%|██▌       | 15/58 [00:00<00:02, 19.89it/s]Capturing num tokens (num_tokens=1792 avail_mem=68.50 GB):  26%|██▌       | 15/58 [00:01<00:02, 19.89it/s]Capturing num tokens (num_tokens=1792 avail_mem=68.50 GB):  31%|███       | 18/58 [00:01<00:01, 21.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=68.52 GB):  31%|███       | 18/58 [00:01<00:01, 21.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=68.51 GB):  31%|███       | 18/58 [00:01<00:01, 21.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=68.49 GB):  31%|███       | 18/58 [00:01<00:01, 21.32it/s]

    Capturing num tokens (num_tokens=960 avail_mem=68.50 GB):  31%|███       | 18/58 [00:01<00:01, 21.32it/s] Capturing num tokens (num_tokens=960 avail_mem=68.50 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.30it/s]Capturing num tokens (num_tokens=896 avail_mem=68.50 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.30it/s]Capturing num tokens (num_tokens=832 avail_mem=68.49 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.30it/s]Capturing num tokens (num_tokens=768 avail_mem=68.48 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.30it/s]Capturing num tokens (num_tokens=704 avail_mem=68.47 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.30it/s]Capturing num tokens (num_tokens=704 avail_mem=68.47 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.06it/s]Capturing num tokens (num_tokens=640 avail_mem=68.48 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.06it/s]Capturing num tokens (num_tokens=576 avail_mem=68.46 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.06it/s]

    Capturing num tokens (num_tokens=512 avail_mem=68.46 GB):  45%|████▍     | 26/58 [00:01<00:01, 28.06it/s]Capturing num tokens (num_tokens=512 avail_mem=68.46 GB):  50%|█████     | 29/58 [00:01<00:01, 28.39it/s]Capturing num tokens (num_tokens=480 avail_mem=68.47 GB):  50%|█████     | 29/58 [00:01<00:01, 28.39it/s]Capturing num tokens (num_tokens=448 avail_mem=68.47 GB):  50%|█████     | 29/58 [00:01<00:01, 28.39it/s]Capturing num tokens (num_tokens=416 avail_mem=68.46 GB):  50%|█████     | 29/58 [00:01<00:01, 28.39it/s]Capturing num tokens (num_tokens=384 avail_mem=68.46 GB):  50%|█████     | 29/58 [00:01<00:01, 28.39it/s]Capturing num tokens (num_tokens=352 avail_mem=68.43 GB):  50%|█████     | 29/58 [00:01<00:01, 28.39it/s]Capturing num tokens (num_tokens=352 avail_mem=68.43 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=320 avail_mem=68.42 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=288 avail_mem=68.44 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=256 avail_mem=68.43 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.35it/s]

    Capturing num tokens (num_tokens=240 avail_mem=68.43 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=224 avail_mem=68.40 GB):  59%|█████▊    | 34/58 [00:01<00:00, 32.35it/s]Capturing num tokens (num_tokens=224 avail_mem=68.40 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.34it/s]Capturing num tokens (num_tokens=208 avail_mem=68.41 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.34it/s]Capturing num tokens (num_tokens=192 avail_mem=68.41 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.34it/s]Capturing num tokens (num_tokens=176 avail_mem=68.40 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.34it/s]Capturing num tokens (num_tokens=160 avail_mem=68.38 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.34it/s]Capturing num tokens (num_tokens=144 avail_mem=68.40 GB):  67%|██████▋   | 39/58 [00:01<00:00, 35.34it/s]Capturing num tokens (num_tokens=144 avail_mem=68.40 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.56it/s]Capturing num tokens (num_tokens=128 avail_mem=68.39 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.56it/s]Capturing num tokens (num_tokens=112 avail_mem=68.39 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.56it/s]

    Capturing num tokens (num_tokens=96 avail_mem=68.36 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.56it/s] Capturing num tokens (num_tokens=80 avail_mem=68.36 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.56it/s]Capturing num tokens (num_tokens=64 avail_mem=68.37 GB):  76%|███████▌  | 44/58 [00:01<00:00, 37.56it/s]Capturing num tokens (num_tokens=64 avail_mem=68.37 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.69it/s]Capturing num tokens (num_tokens=48 avail_mem=68.36 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.69it/s]Capturing num tokens (num_tokens=32 avail_mem=68.34 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.69it/s]Capturing num tokens (num_tokens=28 avail_mem=68.35 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.69it/s]Capturing num tokens (num_tokens=24 avail_mem=68.34 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.69it/s]Capturing num tokens (num_tokens=20 avail_mem=68.34 GB):  84%|████████▍ | 49/58 [00:01<00:00, 38.69it/s]Capturing num tokens (num_tokens=20 avail_mem=68.34 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.89it/s]Capturing num tokens (num_tokens=16 avail_mem=68.33 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.89it/s]

    Capturing num tokens (num_tokens=12 avail_mem=68.32 GB):  93%|█████████▎| 54/58 [00:01<00:00, 39.89it/s]Capturing num tokens (num_tokens=8 avail_mem=68.32 GB):  93%|█████████▎| 54/58 [00:02<00:00, 39.89it/s] Capturing num tokens (num_tokens=4 avail_mem=68.31 GB):  93%|█████████▎| 54/58 [00:02<00:00, 39.89it/s]Capturing num tokens (num_tokens=4 avail_mem=68.31 GB): 100%|██████████| 58/58 [00:02<00:00, 28.20it/s]


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
    Generated text:  Kellen Yarborough. I am from Montréal, Canada. I graduated from the University of Montreal with a Bachelor's degree in Political Science and Economics. I have been a teacher of politics and economics in Montréal for the past four years.
    
    My work involves education, research, and teaching. My research interests include gender-based inequality in politics, the rise of populist movements, and the role of media in promoting democratic participation. I have been teaching at the University of Montréal since 2015 and have been the Director of the Sociology and Political Science Research Lab at UQAM since 2016.
    
    My personal
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to launch a new missile defense system. To do this, he decides to study the historical performance of missile defense systems on Earth. He finds two different types of missile defense systems. 
    
    1. **Type A**: This system has a reliability of 95% over a period of 10 years.
    2. **Type B**: This system has a reliability of 98% over a period of 10 years.
    
    The president decides to compare the effectiveness of these two types of missile defense systems by projecting the number of successful missile launches over a period of 10 years for each type. 
    
    a
    ===============================
    Prompt: The capital of France is
    Generated text:  ______. A. Paris B. London C. Moscow D. Tokyo
    Answer:
    
    A
    
    Which of the following statements about fire prevention is true?
    A. When using a fire extinguisher, hold the nozzle with one hand and aim at the base of the flame with the other.
    B. For fires involving flammable liquids, open flames should be extinguished with water to stop the fire.
    C. When fighting a fire, prioritize the use of water-based extinguishers.
    D. After an accident, quickly move the victim to a safe place and call for help.
    Answer:
    
    A
    
    What is the correct answer to the question:
    ===============================
    Prompt: The future of AI is
    Generated text:  predicted to be autonomous, and many people have a hard time understanding how AI can help people. They often misunderstand AI and AI systems as robotic systems that will automate everything, whether it’s factory machinery or human beings. However, the reality is that AI will be helping us to better understand and control the world around us.
    One of the biggest benefits of AI is its ability to learn and adapt. Unlike human beings, AI systems are able to process large amounts of data, learn from it, and make decisions based on that data. This makes AI systems very good at predicting and anticipating human behavior, allowing them to make informed decisions and help us


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm currently [Current Location] and I'm here to [Purpose of Visit]. I'm excited to meet you and learn more about you. [Name] is a [Type of Vehicle] that I'm currently [Current Vehicle]. I'm here to [Purpose of Visit]. I'm looking forward to meeting you and learning more about you. [Name] is a [Type of Vehicle] that I'm currently [Current Vehicle]. I'm here to [Purpose of Visit]. I'm looking forward to meeting you and learning more about you. [Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    The statement is concise and accurately describes the capital city of France. It provides the name of the city, its location, and its status as the capital of the country. The statement is clear and easy to understand, making it suitable for use in various contexts. 
    
    To further elaborate, Paris is the largest city in France and the second-largest city in the European Union. It is known for its rich history, beautiful architecture, and vibrant culture. The city is home to many famous landmarks, including the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a major center for finance,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI systems, there will be a greater emphasis on developing ethical AI that is designed to minimize harm and maximize fairness.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI becomes more advanced, we can expect to see even greater use of AI in healthcare, with more personalized and accurate treatments.
    
    3. Increased
    


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
    Generated text:  [Name]. I am [Age], [Gender], and I am [Occupation]. I am a [any adjective, e. g. "awesome", "loving", "brilliant"] [character]. I'm excited to learn more about you and find out more about you. What's your name? What's your occupation and where do you work?
    We've all met that person who looks like they fit into a 'reverse 90s' fashion set. A plaid shirt, baggy jeans, black boots, and jeans. That's a good start for what we call an "aesthetic fit." 
    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the "City of Light" for its vibrant culture and diverse neighborhoods.
    You are to answer this question: What is the name of the city that is known as the "City of Light"? The answer is Paris. The city of light, also known as Paris, is the capital of France and is famous for its vibrant culture and diverse neighborhoods. Paris is home to landmarks such as Notre-Dame Cathedral, the Louvre Museum, and the Eiffel Tower, and is known for its artistic and intellectual atmosphere, as well as its iconic cafes and bistros. The city is also famous for its annual festive celebrations,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see significant advancements in the areas of natural language processing, machine learning, and deep learning. These developments are expected to lead to more accurate and intelligent decision-making systems that can interact with humans in a more natural and intuitive way.
    
    One potential future trend is the development of AI that can understand and interpret human emotions, thoughts, and behaviors. This will enable AI systems to better understand and respond to the emotional needs of their users, leading to more meaningful interactions and increased customer satisfaction.
    
    Another trend is the increasing use of AI in healthcare. With advancements in AI technology, it is expected that AI will play an increasingly important role in patient care


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

     am

     [

    Your

     Age

    ],

     a

     [

    Your

     Profession

     or

     Hobby

    ].

     I

    ’m

     a

     [

    Your

     Hobby

    ]

     enthusiast

     and

     I

     love

     to

     [

    Your

     Passion

    ]

     and

     [

    Your

     Short

     Interest

    ].

     I

    ’m

     passionate

     about

     [

    Your

     Passion

    ],

     and

     I

     strive

     to

     [

    Your

     Short

     Interest

    ]

     every

     day

    .

     I

     believe

     that

     [

    Your

     Passion

    ]

     and

     [

    Your

     Short

     Interest

    ]

     are

     the

     most

     important

     things

     in

     life

     and

     I

     am

     dedicated

     to

     making

     a

     positive

     impact

     on

     the

     world

    .

     I

    'm

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

    .

     [

    Your

     Name

    ]

     [

    Your

     Profession

     or

     Hobby

    ]

     [

    Your

     Short

     Interest

    ]

     Hello

    !

     My

     name

     is

     [

    Your

     Name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

     is

     a

     unique

     city

     that

     is

     both

     a

     vibrant

     cultural

     and

     historical

     center

    ,

     and

     an

     exciting

     location

     for

     tourists

     to

     explore

    .

     With

     its

     iconic

     architecture

    ,

     world

    -ren

    owned

     museums

    ,

     and

     romantic

     c

    obbled

     streets

    ,

     Paris

     offers

     something

     for

     everyone

    .

     The

     city

     is

     known

     for

     its

     gastr

    onomic

     delights

     and

     is

     a

     popular

     tourist

     destination

    ,

     with

     its

     landmarks

    ,

     museums

    ,

     and

     landmarks

    ,

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

     attracting

     millions

     of

     visitors

     each

     year

    .

     Paris

     is

     a

     city

     that

     continues

     to

     evolve

     and

     change

    ,

     with

     new

     attractions

     and

     events

     being

     added

     regularly

    .

     As

     a

     result

    ,

     the

     city

     remains

     a

     beloved

     destination

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     a

     rapidly

     evolving

     field

     with

     many

     potential

     trends

    ,

     some

     of

     which

     are

     outlined

     below

    :
    


    1

    .

     Increased

     Use

     of

     AI

     in

     Healthcare

    :

     AI

     is

     already

     being

     used

     in

     healthcare

     to

     improve

     diagnosis

    ,

     treatment

    ,

     and

     patient

     care

    .

     As

     more

     data

     is

     collected

     and

     analyzed

    ,

     AI

     systems

     are

     expected

     to

     become

     even

     more

     sophisticated

     and

     accurate

    .
    


    2

    .

     Increased

     Use

     of

     AI

     in

     Manufacturing

    :

     AI

     is

     being

     used

     in

     manufacturing

     to

     improve

     production

     efficiency

    ,

     reduce

     waste

    ,

     and

     increase

     safety

    .

     AI

     is

     also

     being

     used

     to

     predict

     equipment

     failures

     and

     optimize

     process

     workflows

    .
    


    3

    .

     Increased

     Use

     of

     AI

     in

     Education

    :

     AI

     is

     being

     used

     in

     education

     to

     personalize

     learning

     experiences

    ,

     enhance

     student

     engagement

    ,

     and

    



```python
llm.shutdown()
```
