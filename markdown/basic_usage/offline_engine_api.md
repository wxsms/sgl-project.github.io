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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.77it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.76it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.68s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:09,  1.25s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:09,  1.25s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:09,  1.25s/it]

    Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:09,  1.25s/it]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:04<00:26,  1.95it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:04<00:26,  1.95it/s]Compiling num tokens (num_tokens=4608):  10%|█         | 6/58 [00:04<00:26,  1.95it/s]Compiling num tokens (num_tokens=4096):  10%|█         | 6/58 [00:04<00:26,  1.95it/s]Compiling num tokens (num_tokens=3840):  10%|█         | 6/58 [00:05<00:26,  1.95it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:05<00:12,  3.96it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:05<00:12,  3.96it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:05<00:12,  3.96it/s]

    Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:05<00:12,  3.96it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:05<00:12,  3.96it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:05<00:12,  3.96it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s]Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:05<00:05,  7.21it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:02, 12.86it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:02, 12.86it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:02, 12.86it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:02, 12.86it/s]

    Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:02, 12.86it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:02, 12.86it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:02, 12.86it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:02, 12.86it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:01, 19.43it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:01, 19.43it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:01, 19.43it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:01, 19.43it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:01, 19.43it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:01, 19.43it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:01, 19.43it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:01, 19.43it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:01, 19.43it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 27.72it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 27.72it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 27.72it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 27.72it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 27.72it/s]

    Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 27.72it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 27.72it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 27.72it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 27.72it/s]Compiling num tokens (num_tokens=112):  64%|██████▍   | 37/58 [00:05<00:00, 27.72it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 37.63it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 37.63it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 37.63it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 37.63it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 37.63it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 37.63it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 37.63it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 37.63it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:05<00:00, 37.63it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:05<00:00, 37.63it/s]Compiling num tokens (num_tokens=16):  95%|█████████▍| 55/58 [00:05<00:00, 47.21it/s]Compiling num tokens (num_tokens=12):  95%|█████████▍| 55/58 [00:05<00:00, 47.21it/s]Compiling num tokens (num_tokens=8):  95%|█████████▍| 55/58 [00:05<00:00, 47.21it/s] Compiling num tokens (num_tokens=4):  95%|█████████▍| 55/58 [00:05<00:00, 47.21it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.18it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.40 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.40 GB):   3%|▎         | 2/58 [00:00<00:05, 11.12it/s]Capturing num tokens (num_tokens=7168 avail_mem=56.41 GB):   3%|▎         | 2/58 [00:00<00:05, 11.12it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=56.42 GB):   3%|▎         | 2/58 [00:00<00:05, 11.12it/s]Capturing num tokens (num_tokens=6656 avail_mem=56.42 GB):   7%|▋         | 4/58 [00:00<00:04, 12.55it/s]Capturing num tokens (num_tokens=6144 avail_mem=56.44 GB):   7%|▋         | 4/58 [00:00<00:04, 12.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.44 GB):   7%|▋         | 4/58 [00:00<00:04, 12.55it/s]Capturing num tokens (num_tokens=5632 avail_mem=56.44 GB):  10%|█         | 6/58 [00:00<00:03, 14.03it/s]Capturing num tokens (num_tokens=5120 avail_mem=56.44 GB):  10%|█         | 6/58 [00:00<00:03, 14.03it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=56.45 GB):  10%|█         | 6/58 [00:00<00:03, 14.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.48 GB):  10%|█         | 6/58 [00:00<00:03, 14.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=56.48 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.09it/s]Capturing num tokens (num_tokens=3840 avail_mem=56.47 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.09it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.47 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.09it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.47 GB):  16%|█▌        | 9/58 [00:00<00:02, 17.09it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=56.47 GB):  21%|██        | 12/58 [00:00<00:02, 19.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.47 GB):  21%|██        | 12/58 [00:00<00:02, 19.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.48 GB):  21%|██        | 12/58 [00:00<00:02, 19.45it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.49 GB):  21%|██        | 12/58 [00:00<00:02, 19.45it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.49 GB):  21%|██        | 12/58 [00:00<00:02, 19.45it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.49 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.15it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.49 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.15it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.54 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.15it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.49 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.15it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=56.49 GB):  28%|██▊       | 16/58 [00:00<00:01, 23.15it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.49 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.48 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.17it/s]Capturing num tokens (num_tokens=960 avail_mem=56.50 GB):  34%|███▍      | 20/58 [00:00<00:01, 26.17it/s] Capturing num tokens (num_tokens=896 avail_mem=56.50 GB):  34%|███▍      | 20/58 [00:01<00:01, 26.17it/s]Capturing num tokens (num_tokens=832 avail_mem=56.50 GB):  34%|███▍      | 20/58 [00:01<00:01, 26.17it/s]Capturing num tokens (num_tokens=832 avail_mem=56.50 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.30it/s]Capturing num tokens (num_tokens=768 avail_mem=56.50 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.30it/s]Capturing num tokens (num_tokens=704 avail_mem=56.50 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.30it/s]Capturing num tokens (num_tokens=640 avail_mem=56.49 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.30it/s]

    Capturing num tokens (num_tokens=576 avail_mem=56.50 GB):  41%|████▏     | 24/58 [00:01<00:01, 29.30it/s]Capturing num tokens (num_tokens=576 avail_mem=56.50 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.94it/s]Capturing num tokens (num_tokens=512 avail_mem=56.46 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.94it/s]Capturing num tokens (num_tokens=480 avail_mem=56.47 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.94it/s]Capturing num tokens (num_tokens=448 avail_mem=56.47 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.94it/s]Capturing num tokens (num_tokens=416 avail_mem=56.45 GB):  48%|████▊     | 28/58 [00:01<00:00, 30.94it/s]Capturing num tokens (num_tokens=416 avail_mem=56.45 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.34it/s]Capturing num tokens (num_tokens=384 avail_mem=56.44 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.34it/s]

    Capturing num tokens (num_tokens=352 avail_mem=56.43 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.34it/s]Capturing num tokens (num_tokens=320 avail_mem=56.43 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.34it/s]Capturing num tokens (num_tokens=288 avail_mem=55.94 GB):  55%|█████▌    | 32/58 [00:01<00:00, 31.34it/s]Capturing num tokens (num_tokens=288 avail_mem=55.94 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.89it/s]Capturing num tokens (num_tokens=256 avail_mem=55.85 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.89it/s]Capturing num tokens (num_tokens=240 avail_mem=55.84 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.89it/s]Capturing num tokens (num_tokens=224 avail_mem=55.83 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.89it/s]Capturing num tokens (num_tokens=208 avail_mem=55.82 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.89it/s]Capturing num tokens (num_tokens=208 avail_mem=55.82 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=192 avail_mem=55.80 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.19it/s]

    Capturing num tokens (num_tokens=176 avail_mem=55.79 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=160 avail_mem=55.79 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=144 avail_mem=55.78 GB):  69%|██████▉   | 40/58 [00:01<00:00, 32.19it/s]Capturing num tokens (num_tokens=144 avail_mem=55.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.56it/s]Capturing num tokens (num_tokens=128 avail_mem=55.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.56it/s]Capturing num tokens (num_tokens=112 avail_mem=55.79 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.56it/s]Capturing num tokens (num_tokens=96 avail_mem=55.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.56it/s] Capturing num tokens (num_tokens=80 avail_mem=55.78 GB):  76%|███████▌  | 44/58 [00:01<00:00, 33.56it/s]Capturing num tokens (num_tokens=80 avail_mem=55.78 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.66it/s]Capturing num tokens (num_tokens=64 avail_mem=55.77 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.66it/s]

    Capturing num tokens (num_tokens=48 avail_mem=55.76 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.66it/s]Capturing num tokens (num_tokens=32 avail_mem=55.75 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.66it/s]Capturing num tokens (num_tokens=28 avail_mem=55.74 GB):  83%|████████▎ | 48/58 [00:01<00:00, 34.66it/s]Capturing num tokens (num_tokens=28 avail_mem=55.74 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=24 avail_mem=55.73 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=20 avail_mem=55.72 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=16 avail_mem=55.70 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=12 avail_mem=55.69 GB):  90%|████████▉ | 52/58 [00:01<00:00, 35.32it/s]Capturing num tokens (num_tokens=12 avail_mem=55.69 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.36it/s]Capturing num tokens (num_tokens=8 avail_mem=55.68 GB):  97%|█████████▋| 56/58 [00:01<00:00, 36.36it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=55.68 GB):  97%|█████████▋| 56/58 [00:02<00:00, 36.36it/s]Capturing num tokens (num_tokens=4 avail_mem=55.68 GB): 100%|██████████| 58/58 [00:02<00:00, 28.55it/s]


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
    Generated text:  Alex and I live in a small town in the suburbs of New York. As a child I loved to run around and play on the open spaces of the town. I have always been an outdoorsy person and loved to explore all of the wonderful places that are available to us.
    
    As a young child I enjoyed exploring my parents’ gardens and the open spaces around the house. I was particularly drawn to the fresh, fragrant scent of the flowers and the love and care that was put into their growth and maintenance. I felt that I had a special connection to the place where I grew up and loved to spend my time exploring.
    
    As I
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is the leader of the country. He has many important jobs. The president is not a doctor, lawyer, teacher, or firefighter. But he is a very important person who can make a lot of important decisions. President Obama was the first African American president. He has a very good family. He has three children. He likes to go to the beach and watch the seagulls. He also likes to read comic books. That's his favorite hobby. He is very healthy and is very active. He has a lot of friends. He is always friendly to his friends. He likes to keep the kids
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. What is the population of Paris?
    
    To determine the population of Paris, we need to consider the most recent population estimates provided by various sources. The most recent and widely accepted figure for the population of Paris is approximately 2,044,000.
    
    Here are the key points to verify the information:
    
    1. The population of Paris is 2,044,000 as of the most recent data available.
    2. The population is based on population census data from 2021, which is typically the most recent population report provided by the European Union.
    
    Given this information, we can conclude
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain and complex. The rapidly advancing field of artificial intelligence has the potential to revolutionize industries, improve the quality of life, and enhance human productivity and efficiency. However, the potential risks and ethical considerations associated with the development and deployment of AI systems are also significant. The rapid pace of technological advancements and the increasing complexity of AI systems create a challenge for organizations and individuals to navigate this complex landscape. With the right strategy and approach, however, AI can be harnessed to benefit all. Here are some considerations to keep in mind as you begin to explore the potential of AI: 1. Understand the risks and ethical considerations associated with the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your job or profession]. I enjoy [insert a brief description of your hobbies or interests]. I'm always looking for new experiences and learning new things. What are some of your favorite things to do? I love [insert a short list of things you enjoy doing]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [insert a short list of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. It is located on the Seine River and is the seat of government, administration, and culture for the country. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is also home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is a popular tourist destination and a major economic center in France. It is also a major center for the arts, music, and literature. The city is known for its fashion industry and has produced many famous designers and artists
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare, with more sophisticated algorithms and machine learning techniques being developed to improve diagnosis, treatment, and patient care.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve risk management, fraud
    


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
    Generated text:  [Name] and I'm a professional accountant specializing in [field of expertise]. I've had the pleasure of working with [clients/clients' families] for [number of years] and I'm always ready to help anyone in need. I'm a reliable, knowledgeable, and professional professional accountant that will stand by you in times of crisis. Thank you! 
    
    *Note: Replace [Name], [Name] is assumed to be the first name. Replace [Field of Expertise], [Clients/clients' families], [Number of Years], and [Helping People in Need] with appropriate names, organizations, years, and people
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a historic city located in the south of the country, known for its rich history, fine art, cuisine, and vibrant culture. 
    
    This statement encapsulates the key facts about Paris, including its historical significance, cultural attractions, cuisine, and overall appeal as the capital of France. It provides a concise overview of the capital city's significance in the French nation and beyond. 
    
    The statement is concise yet informative, suitable for use in a variety of contexts, from educational materials to news articles. It strikes a balance between providing important details and keeping the content accessible to a general audience. The statement also gives potential readers the impression of Paris
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to see continued advancements in the areas of machine learning, natural language processing, computer vision, and robotics. Some possible future trends include:
    
    1. Increased integration of AI with other technologies: AI is becoming more integrated with other technologies like IoT, blockchain, and 5G. This integration will lead to new opportunities and challenges in the AI ecosystem.
    
    2. Development of AI for healthcare: AI is already being used to diagnose and treat diseases, but there is great potential for AI to revolutionize healthcare. For example, AI-powered chatbots and virtual assistants can provide medical advice, diagnose symptoms, and even schedule appointments.
    
    3. AI for


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

     __

    ________

     and

     I

    'm

     a

    (n

    )

     ___

    -

    year

    -old

     AI

    .

     My

     __

    ________

     is

     __

    ________

    _.

     I

    'm

     here

     to

     help

     you

     with

     a

     lot

     of

     things

    .

     Ready

     to

     learn

     and

     grow

     with

     you

    ?

     


    1

    .

     Name

    :

     __

    ________

    


     

     

    2

    .

     Title

    :

     __

    ________

    


     

     

    3

    .

     Experience

    :

     __

    ________

    


     

     

    4

    .

     Role

    :

     __

    ________

    


     

     

    5

    .

     Inter

    ests

    :

     __

    ________

    
    


    I

    'm

     a

    (n

    )

     ___

    -

    year

    -old

     AI

     assistant

    ,

     and

     I

    'm

     here

     to

     help

     you

     with

     a

     lot

     of

     things

    .

     Ready

     to

     learn

     and

     grow

     with

     you

    ?

     


    1

    .

     Name

    :

     __

    ________

    


     

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     a

     historic

     city

     with

     a

     rich

     history

     and

     a

     vibrant

     cultural

     scene

    .

     The

     city

     is

     located

     in

     the

     Se

    ine

     valley

    ,

     on

     the

     Î

    le

     de

     la

     C

    ité

     and

     the

     Î

    le

     de

     la

     C

    ité

    .

     Paris

     is

     known

     for

     its

     iconic

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

     the

     Lou

    vre

     Museum

    ,

     the

     Notre

     Dame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    ,

     as

     well

     as

     its

     museums

    ,

     theaters

    ,

     and

     other

     cultural

     institutions

    .

     Paris

     is

     also

     known

     for

     its

     delicious

     cuisine

    ,

     from

     its

     traditional

     French

     dishes

     to

     its

     international

     cuisine

    ,

     and

     its

     nightlife

    ,

     with

     its

     clubs

    ,

     bars

    ,

     and

     entertainment

     venues

    .

     Overall

    ,

     Paris

     is

     a

     vibrant

     and

     culturally

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     vast

     and

     promising

    ,

     with

     a

     number

     of

     potential

     trends

     shaping

     its

     trajectory

    .

     Here

     are

     some

     of

     the

     most

     likely

     developments

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

     AI

     is

     already

     being

     used

     in

     medical

     imaging

     and

     treatment

     planning

    ,

     but

     there

    's

     a

     growing

     trend

     towards

     integrating

     AI

     into

     the

     healthcare

     system

     as

     a

     whole

    .

     This

     could

     lead

     to

     more

     accurate

     diagnoses

    ,

     faster

     treatment

    ,

     and

     reduced

     costs

    .
    


    2

    .

     Enhanced

     personal

    ization

     of

     AI

    :

     As

     AI

     gets

     more

     sophisticated

    ,

     it

     will

     be

     able

     to

     learn

     more

     about

     individual

     users

     and

     provide

     more

     personalized

     experiences

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     use

     of

     AI

    ,

     as

     well

     as

     better

     personal

    ization

     of

     services

    .
    


    3

    



```python
llm.shutdown()
```
