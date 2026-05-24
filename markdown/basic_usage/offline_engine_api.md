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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.50it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:41,  3.89s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<00:57,  1.05s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<00:57,  1.05s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<00:57,  1.05s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:28,  1.84it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:28,  1.84it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:28,  1.84it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:28,  1.84it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:14,  3.53it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:14,  3.53it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:14,  3.53it/s]

    Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:14,  3.53it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:08,  5.67it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:04,  8.96it/s]

    Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:04,  8.96it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:02, 13.68it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:02, 13.68it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:01, 19.05it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:01, 19.05it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:01, 19.05it/s]

    Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:01, 19.05it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:01, 19.05it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:01, 19.05it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:01, 19.05it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 26.15it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 26.15it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 26.15it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 26.15it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 26.15it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 26.15it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 26.15it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 26.15it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 26.15it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 36.34it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 36.34it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 36.34it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 36.34it/s]

    Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 36.34it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:05<00:00, 36.34it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:05<00:00, 36.34it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:05<00:00, 36.34it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:05<00:00, 42.32it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:05<00:00, 42.32it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:05<00:00, 42.32it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:05<00:00, 42.32it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:05<00:00, 42.32it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:05<00:00, 42.32it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:05<00:00, 42.32it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:05<00:00, 42.32it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 48.27it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 48.27it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 48.27it/s]

    Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 48.27it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 48.27it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 48.27it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=54.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=54.34 GB):   3%|▎         | 2/58 [00:00<00:04, 13.67it/s]Capturing num tokens (num_tokens=7168 avail_mem=54.34 GB):   3%|▎         | 2/58 [00:00<00:04, 13.67it/s]Capturing num tokens (num_tokens=6656 avail_mem=54.34 GB):   3%|▎         | 2/58 [00:00<00:04, 13.67it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=54.34 GB):   3%|▎         | 2/58 [00:00<00:04, 13.67it/s]Capturing num tokens (num_tokens=6144 avail_mem=54.34 GB):   9%|▊         | 5/58 [00:00<00:02, 19.45it/s]Capturing num tokens (num_tokens=5632 avail_mem=54.33 GB):   9%|▊         | 5/58 [00:00<00:02, 19.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.32 GB):   9%|▊         | 5/58 [00:00<00:02, 19.45it/s]Capturing num tokens (num_tokens=5120 avail_mem=54.32 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.55it/s]Capturing num tokens (num_tokens=4608 avail_mem=54.32 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.55it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=54.32 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.55it/s]Capturing num tokens (num_tokens=4096 avail_mem=54.32 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.52it/s]Capturing num tokens (num_tokens=3840 avail_mem=54.31 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.52it/s]Capturing num tokens (num_tokens=3584 avail_mem=54.31 GB):  16%|█▌        | 9/58 [00:00<00:03, 14.52it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=54.31 GB):  19%|█▉        | 11/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=3328 avail_mem=54.31 GB):  19%|█▉        | 11/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.30 GB):  19%|█▉        | 11/58 [00:00<00:03, 14.13it/s]Capturing num tokens (num_tokens=3072 avail_mem=54.30 GB):  22%|██▏       | 13/58 [00:00<00:03, 14.41it/s]Capturing num tokens (num_tokens=2816 avail_mem=54.30 GB):  22%|██▏       | 13/58 [00:00<00:03, 14.41it/s]Capturing num tokens (num_tokens=2560 avail_mem=54.30 GB):  22%|██▏       | 13/58 [00:00<00:03, 14.41it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=54.30 GB):  22%|██▏       | 13/58 [00:00<00:03, 14.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.29 GB):  22%|██▏       | 13/58 [00:00<00:03, 14.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=54.29 GB):  29%|██▉       | 17/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=54.29 GB):  29%|██▉       | 17/58 [00:00<00:02, 20.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=54.29 GB):  29%|██▉       | 17/58 [00:01<00:02, 20.12it/s]Capturing num tokens (num_tokens=1280 avail_mem=54.28 GB):  29%|██▉       | 17/58 [00:01<00:02, 20.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=54.26 GB):  29%|██▉       | 17/58 [00:01<00:02, 20.12it/s]Capturing num tokens (num_tokens=960 avail_mem=54.28 GB):  29%|██▉       | 17/58 [00:01<00:02, 20.12it/s] Capturing num tokens (num_tokens=960 avail_mem=54.28 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.48it/s]Capturing num tokens (num_tokens=896 avail_mem=54.28 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.48it/s]Capturing num tokens (num_tokens=832 avail_mem=54.27 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.48it/s]Capturing num tokens (num_tokens=768 avail_mem=54.27 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.48it/s]

    Capturing num tokens (num_tokens=704 avail_mem=54.27 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.48it/s]Capturing num tokens (num_tokens=640 avail_mem=54.12 GB):  38%|███▊      | 22/58 [00:01<00:01, 27.48it/s]Capturing num tokens (num_tokens=640 avail_mem=54.12 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=576 avail_mem=54.12 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=512 avail_mem=54.10 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=480 avail_mem=54.12 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=448 avail_mem=54.08 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=416 avail_mem=53.75 GB):  47%|████▋     | 27/58 [00:01<00:00, 32.93it/s]Capturing num tokens (num_tokens=416 avail_mem=53.75 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=384 avail_mem=53.75 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.98it/s]

    Capturing num tokens (num_tokens=352 avail_mem=53.56 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=320 avail_mem=53.56 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=288 avail_mem=53.56 GB):  55%|█████▌    | 32/58 [00:01<00:00, 34.98it/s]Capturing num tokens (num_tokens=288 avail_mem=53.56 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.09it/s]Capturing num tokens (num_tokens=256 avail_mem=53.56 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.09it/s]Capturing num tokens (num_tokens=240 avail_mem=53.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.09it/s]

    Capturing num tokens (num_tokens=224 avail_mem=53.55 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.09it/s]Capturing num tokens (num_tokens=208 avail_mem=53.54 GB):  62%|██████▏   | 36/58 [00:01<00:00, 29.09it/s]Capturing num tokens (num_tokens=208 avail_mem=53.54 GB):  69%|██████▉   | 40/58 [00:01<00:00, 27.52it/s]Capturing num tokens (num_tokens=192 avail_mem=53.54 GB):  69%|██████▉   | 40/58 [00:01<00:00, 27.52it/s]Capturing num tokens (num_tokens=176 avail_mem=53.54 GB):  69%|██████▉   | 40/58 [00:01<00:00, 27.52it/s]Capturing num tokens (num_tokens=160 avail_mem=53.54 GB):  69%|██████▉   | 40/58 [00:01<00:00, 27.52it/s]

    Capturing num tokens (num_tokens=160 avail_mem=53.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.67it/s]Capturing num tokens (num_tokens=144 avail_mem=53.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.67it/s]Capturing num tokens (num_tokens=128 avail_mem=53.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.67it/s]Capturing num tokens (num_tokens=112 avail_mem=53.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 26.67it/s]Capturing num tokens (num_tokens=112 avail_mem=53.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 26.29it/s]Capturing num tokens (num_tokens=96 avail_mem=53.53 GB):  79%|███████▉  | 46/58 [00:01<00:00, 26.29it/s] Capturing num tokens (num_tokens=80 avail_mem=53.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 26.29it/s]Capturing num tokens (num_tokens=64 avail_mem=53.52 GB):  79%|███████▉  | 46/58 [00:01<00:00, 26.29it/s]Capturing num tokens (num_tokens=48 avail_mem=53.51 GB):  79%|███████▉  | 46/58 [00:01<00:00, 26.29it/s]

    Capturing num tokens (num_tokens=48 avail_mem=53.51 GB):  86%|████████▌ | 50/58 [00:02<00:00, 23.61it/s]Capturing num tokens (num_tokens=32 avail_mem=72.80 GB):  86%|████████▌ | 50/58 [00:02<00:00, 23.61it/s]Capturing num tokens (num_tokens=28 avail_mem=72.79 GB):  86%|████████▌ | 50/58 [00:02<00:00, 23.61it/s]Capturing num tokens (num_tokens=24 avail_mem=72.79 GB):  86%|████████▌ | 50/58 [00:02<00:00, 23.61it/s]Capturing num tokens (num_tokens=24 avail_mem=72.79 GB):  91%|█████████▏| 53/58 [00:02<00:00, 22.41it/s]Capturing num tokens (num_tokens=20 avail_mem=72.78 GB):  91%|█████████▏| 53/58 [00:02<00:00, 22.41it/s]Capturing num tokens (num_tokens=16 avail_mem=72.78 GB):  91%|█████████▏| 53/58 [00:02<00:00, 22.41it/s]Capturing num tokens (num_tokens=12 avail_mem=72.78 GB):  91%|█████████▏| 53/58 [00:02<00:00, 22.41it/s]

    Capturing num tokens (num_tokens=8 avail_mem=72.77 GB):  91%|█████████▏| 53/58 [00:02<00:00, 22.41it/s] Capturing num tokens (num_tokens=4 avail_mem=72.77 GB):  91%|█████████▏| 53/58 [00:02<00:00, 22.41it/s]Capturing num tokens (num_tokens=4 avail_mem=72.77 GB): 100%|██████████| 58/58 [00:02<00:00, 28.12it/s]Capturing num tokens (num_tokens=4 avail_mem=72.77 GB): 100%|██████████| 58/58 [00:02<00:00, 24.33it/s]


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
    Generated text:  Nicole and I am a registered nurse who has been working at the local high school for the past 6 years. As a new nurse, I am excited to share my experiences and expertise in nursing and care delivery with others. I am here to connect with other nurses and healthcare providers who share my passion for caring for others and helping them feel better. I hope that we can learn from each other and share resources and resources to improve the quality of care we provide. I am excited to help others find the resources and support they need to succeed in their lives. Thank you for considering my request. What is your role in the nursing community and
    ===============================
    Prompt: The president of the United States is
    Generated text:  20 years older than the president of Brazil. The president of Brazil is 25 years younger than the president of the United States. If the president of the United States is currently 50 years old, how old is the president of Brazil?
    To determine the age of the president of Brazil, we start by defining the variables and using the information given in the problem.
    
    1. Let \( U \) represent the age of the president of the United States.
    2. Let \( B \) represent the age of the president of Brazil.
    
    From the problem, we know:
    - The president of the United States is currently 
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Rome
    D. Berlin
    The correct answer is A. Paris. Paris is the capital of France and is known for its iconic landmarks such as the Eiffel Tower and the Notre-Dame Cathedral. The other options, London, Rome, and Berlin, are not capitals of France. London is the capital of England, Rome is the capital of Italy, and Berlin is the capital of Germany. The Eiffel Tower and the Notre-Dame Cathedral are located in Paris, France. However, the Eiffel Tower is located in Paris, France. Therefore, the correct answer
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the people who will be executing it, as it is fundamentally an ethical issue. For example, if we are to create a system that can accurately predict the next earthquake or the next airplane crash, then we need to ensure that the system can not only be accurate in predicting the events, but also that it can be reliable, and should not be manipulated to harm anyone. Similarly, to create a system that can execute all 100 million security checks a second, we need to ensure that the system can accurately and reliably carry out its tasks, and that it does not harm anyone.
    This is also true for other


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] [Occupation]. I'm a [Skill or特长] that I've honed over the years. I'm always looking for new challenges and opportunities to grow and improve. I'm always eager to learn and adapt to new situations. I'm a [Favorite hobby or activity] that I enjoy doing. I'm always looking for ways to make my life more interesting and enjoyable. I'm a [Personality trait] that I bring to every interaction I have. I'm always ready to help and support others. I'm a [Motivation] that drives me
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and Louvre Museum. Paris is also known for its cuisine, fashion, and art scene. The city is home to many world-renowned museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. Paris is a popular tourist destination and a major economic center in France. It is also home to many important political and cultural institutions, including the French Parliament and the French Academy of Sciences
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing it to perform tasks that are currently only possible with human expertise. This could lead to a more human-like AI that can make decisions and take action based on human values and ethics.
    
    2. Enhanced privacy and security: As AI becomes more advanced, there will be an increased need for privacy and security measures to protect the data that is generated and used by
    


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
    Generated text:  [name] and I am a passionate and curious individual who loves to explore the world around me. I am a highly educated and well-traveled individual who values knowledge and learning. I love to travel and I enjoy discovering new things and learning about different cultures and histories. I am also a curious and independent person who likes to challenge myself and try new things. In short, I am a curious and adventurous individual who enjoys learning and experiencing new things. I am excited to meet you and share my knowledge and experiences with you. 
    My name is [name], and I am [age]. I am [occupation], and I enjoy [activities
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its rich history, architecture, and vibrant culture. The city is home to numerous landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Louvre Museum. Paris is also a popular tourist destination, attracting millions of visitors annually. With its narrow streets, colorful architecture, and lively nightlife, Paris is a destination in itself. The city's history, from ancient Romans to the French Revolution, leaves a lasting impression on visitors and locals alike. Paris is the cultural, political, and economic center of France and a key hub for international trade. Paris's skyline is dotted with
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  promising and exciting, with many possibilities and opportunities. Here are some of the potential future trends in artificial intelligence:
    
    1. Increased use of AI in healthcare: AI-powered medical assistants, chatbots, and virtual assistants are becoming increasingly common in healthcare. These assistants can help with patient documentation, scheduling appointments, and providing emotional support, which can improve patient satisfaction and reduce healthcare costs.
    
    2. Personalized medicine: AI can be used to analyze vast amounts of medical data to identify patterns and predict outcomes, which can lead to more effective and personalized treatments. For example, AI algorithms can predict which patients are likely to respond to certain medications or treatments,


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

    /an

     [

    Title

     or

     profession

    ]

     who

    's

     passionate

     about

     [

    What

     exc

    ites

     you

     about

     your

     career

    ].

     I

    'm

     always

     looking

     for

     new

     and

     exciting

     challenges

     to

     overcome

     and

     am

     always

     eager

     to

     learn

     from

     my

     colleagues

     and

     the

     world

     around

     me

    .

     I

    'm

     very

     down

    -to

    -earth

     and

     approach

    able

    ,

     always

     ready

     to

     help

     others

     and

     make

     a

     difference

     in

     their

     lives

    .

     I

    'm

     a

     team

     player

    ,

     and

     I

     believe

     in

     unity

     and

     collaboration

    .

     My

     motto

     is

     "

    Always

     be

     the

     best

     version

     of

     yourself

    ,"

     and

     I

    'm

     constantly

     pushing

     myself

     to

     improve

     and

     grow

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

     what

     you

     have

     to

     offer

    .
    


    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .


    Paris

    ,

     the

     vibrant

     and

     iconic

     city

    ,

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     culture

    .

     The

     city

     is

     also

     home

     to

     one

     of

     the

     world

    's

     most

     famous

     art

     museums

    ,

     the

     Lou

    vre

    ,

     and

     has

     a

     world

    -ren

    owned

     culinary

     scene

    .

     France

    's

     capital

     is

     a

     bustling

     hub

     for

     politics

    ,

     fashion

    ,

     and

     entertainment

    ,

     and

     its

     annual

     Paris

    ian

     Carnival

     is

     a

     major

     event

     in

     the

     city

    's

     calendar

    .

     Paris

     is

     also

     a

     popular

     tourist

     destination

    ,

     attracting

     millions

     of

     visitors

     each

     year

     for

     its

     stunning

     views

    ,

     delicious

     cuisine

    ,

     and

     lively

     atmosphere

    .

     The

     capital

     of

     France

     is

     a

     cultural

     and

     historical

     gem

     that

     has

     played

     a

     significant

     role

     in

     French

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     driven

     by

     a

     combination

     of

     technological

     advancements

    ,

     changes

     in

     the

     way

     we

     interact

     with

     machines

    ,

     and

     shifts

     in

     our

     understanding

     of

     human

     intelligence

    .

     Here

     are

     some

     possible

     trends

     that

     could

     shape

     the

     future

     of

     AI

    :
    


    1

    .

     Increased

     automation

    :

     As

     AI

     continues

     to

     improve

     and

     become

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     are

     likely

     to

     see

     an

     increase

     in

     automation

    ,

     where

     machines

     perform

     tasks

     that

     were

     previously

     the

     domain

     of

     humans

    .

     This

     could

     lead

     to

     job

     displacement

    ,

     but

     it

     could

     also

     create

     new

     opportunities

     for

     workers

     who

     are

     able

     to

     adapt

     to

     new

     roles

    .
    


    2

    .

     Enhanced

     human

    -A

    I

     collaboration

    :

     With

     the

     increasing

     integration

     of

     AI

     into

     our

     daily

     lives

    ,

     there

    



```python
llm.shutdown()
```
