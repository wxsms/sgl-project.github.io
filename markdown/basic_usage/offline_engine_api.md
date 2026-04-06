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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.09it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.07it/s]


    2026-04-06 07:41:59,380 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-06 07:41:59] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:34,  2.72s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.32it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:02<00:07,  6.11it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:03<00:07,  6.11it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:03<00:07,  6.11it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:03<00:07,  6.11it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:03<00:03, 12.40it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:03<00:03, 12.40it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:03<00:03, 12.40it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:03<00:03, 12.40it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:03<00:03, 12.40it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:03, 12.40it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:03, 12.40it/s]

    Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:03, 12.40it/s]Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=480):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=448):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=416):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=384):  45%|████▍     | 26/58 [00:03<00:01, 18.47it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:03<00:00, 25.10it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:03<00:00, 25.10it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:03<00:00, 25.10it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:03<00:00, 25.10it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:03<00:00, 25.10it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:03<00:00, 25.10it/s]

    Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:03<00:00, 25.10it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.17it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.17it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.17it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.17it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.17it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.17it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.17it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 35.14it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 35.14it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 35.14it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 35.14it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 35.14it/s]

    Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 35.14it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:03<00:00, 35.14it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:03<00:00, 39.32it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:03<00:00, 39.32it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:03<00:00, 39.32it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:03<00:00, 39.32it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:03<00:00, 39.32it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:03<00:00, 39.32it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:03<00:00, 39.32it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:03<00:00, 39.32it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.41 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.38 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.97it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.37 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.37 GB):   9%|▊         | 5/58 [00:00<00:02, 22.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.90it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.90it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.90it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.90it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.90it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.49it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.74it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.74it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.31 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.74it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.74it/s] Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.74it/s]Capturing num tokens (num_tokens=832 avail_mem=137.32 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.74it/s]Capturing num tokens (num_tokens=832 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.63it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.63it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.63it/s]Capturing num tokens (num_tokens=640 avail_mem=137.31 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.63it/s]Capturing num tokens (num_tokens=576 avail_mem=137.31 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.63it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.63it/s]Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  50%|█████     | 29/58 [00:00<00:00, 41.58it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  50%|█████     | 29/58 [00:00<00:00, 41.58it/s]Capturing num tokens (num_tokens=448 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 41.58it/s]

    Capturing num tokens (num_tokens=416 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 41.58it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  50%|█████     | 29/58 [00:00<00:00, 41.58it/s]Capturing num tokens (num_tokens=352 avail_mem=137.30 GB):  50%|█████     | 29/58 [00:00<00:00, 41.58it/s]Capturing num tokens (num_tokens=352 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.97it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.97it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.97it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.97it/s]Capturing num tokens (num_tokens=240 avail_mem=137.29 GB):  59%|█████▊    | 34/58 [00:00<00:00, 42.97it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 42.97it/s]Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.70it/s]

    Capturing num tokens (num_tokens=176 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=144 avail_mem=137.27 GB):  67%|██████▋   | 39/58 [00:01<00:00, 43.70it/s]Capturing num tokens (num_tokens=144 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=128 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=96 avail_mem=137.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s] Capturing num tokens (num_tokens=80 avail_mem=137.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.50it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.69it/s]

    Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=24 avail_mem=137.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  84%|████████▍ | 49/58 [00:01<00:00, 42.69it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.97it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.97it/s]

    Capturing num tokens (num_tokens=12 avail_mem=137.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.97it/s]Capturing num tokens (num_tokens=8 avail_mem=137.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.97it/s] Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  93%|█████████▎| 54/58 [00:01<00:00, 32.97it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:01<00:00, 36.53it/s]


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
    Generated text:  Max and I am a professional lawyer who works in a law firm in the United States. My goal is to provide quality legal advice and representation to individuals who are facing a range of legal issues. My background includes significant experience in various areas of law, including personal injury, family law, and intellectual property law.
    As a lawyer, I am always committed to providing my clients with the most effective and efficient legal representation possible. My clients are treated with respect and confidentiality, and I strive to work with my clients to come up with the best possible solution to their legal problems. I am committed to providing ongoing education and development for my clients to help
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man. Which of the following, if true, would most effectively undermine the claim that the president is a man? I. One in every ten men are born blind. II. The average American male is 5 feet 10 inches tall. III. All men have brown eyes. IV. The president’s hair is white.
    I and II. II and III. II and IV. I and III.
    I and IV.
    I, II, and IV.
    II and IV.
    Answer:
    
    C
    
    Which of the following statements about children's rights is true? ① Children should enjoy the rights stipulated in the constitution
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, and it is located in which continent?
    A. North America
    B. Asia
    C. Africa
    D. Europe
    Answer:
    
    D
    
    What is the primary function of a converter transformer in a power system?
    A. To step up voltage
    B. To step down voltage
    C. To step up and down voltage
    D. To step down and step up voltage
    Answer:
    
    C
    
    What are the two main components of a power supply system?
    A. Power source and distribution system
    B. Power source and load
    C. Power source and power grid
    D. Load and power grid
    Answer:
    
    C
    
    
    ===============================
    Prompt: The future of AI is
    Generated text:  developing rapidly, and developers are always looking to improve and innovate in this field. With the continuous advancements in AI, there are numerous job roles and opportunities that can be filled with the help of AI technologies. However, to ensure that the jobs become accessible to all people, companies must also ensure that they are welcoming to and inclusive of diverse backgrounds. Here are some steps that companies can take to make their job roles and opportunities more accessible to all people:
    
    1. Diversity and Inclusion: Companies must ensure that they have an inclusive and diverse workforce. This means hiring people from all backgrounds, cultures, and backgrounds, and providing equal opportunities for all


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Type of Vehicle] with [Number] wheels. I'm a [Type of Vehicle] with [Number] wheels. I'm a [Type of Vehicle] with [Number] wheels. I'm a [Type of Vehicle] with [Number] wheels. I'm a [Type of Vehicle] with [Number] wheels. I'm a [Type of Vehicle] with [Number] wheels. I'm a [Type of Vehicle] with [Number] wheels. I'm a [Type of Vehicle] with [Number] wheels.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the annual Eiffel Tower Festival. It is also the seat of the French government and the largest city in the country. Paris is a cultural and historical center with many museums, art galleries, and landmarks. The city is also known for its cuisine, fashion, and music. It is a popular tourist destination and a major economic hub. Paris is a city of contrasts, with its rich history and modernity. It is a city that has been shaped by the French Revolution, the French Revolution, and the French Revolution. It is a city that has been shaped by the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more sophisticated and personalized AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased scrutiny of AI systems in the public eye.
    
    3. Increased use of AI in healthcare: AI is already being
    


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
    Generated text:  ___________ and I’m ___________ years old.
    
    Hello, my name is [Name] and I’m [Age] years old.
    
    If you’d like to know more about me, feel free to ask and I’ll be happy to provide a detailed answer. Here’s what I’m really good at:
    
    1. [List one or more skills or abilities you're particularly strong at]
    2. [List any other qualities or qualities that make you stand out]
    
    I hope you enjoy learning more about me! Let me know if there’s anything specific you’d like to know or if you have any questions. 
    
    [Your Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. Paris is the largest city in France, home to approximately 2.3 million inhabitants. It is known for its rich history, vibrant culture, and picturesque architecture. The city is also a UNESCO World Heritage Site and a significant commercial hub for Europe. Paris is a major center for French culture, politics, and economy. It is home to iconic landmarks such as Notre-Dame Cathedral, the Louvre Museum, and the Eiffel Tower. Paris is also a popular destination for tourists and business travelers alike. With its stunning architecture, diverse culture, and friendly residents, Paris is a city of contrasts and endless possibilities. 
    
    -
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  uncertain, but here are some possible trends that are likely to shape its development:
    
    1. Increased AI integration with other technologies: AI will become more integrated with other technologies, such as the Internet of Things (IoT) and sensors. This will enable more accurate and efficient use of AI and provide new applications and capabilities.
    
    2. AI to become more human-like: AI is likely to evolve in ways that make it more like the human mind, with more emotional and social intelligence. This could lead to more empathetic AI and a more nuanced understanding of human behavior.
    
    3. AI is expected to become more ubiquitous: The adoption of AI will


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

    ...

     
    


    That

    's

     all

     you

     need

    .

     You

     have

     no

     idea

     what

     that

     means

    ,

     but

     I

     can

     do

     that

    .

     
    


    Just

     keep

     it

     short

     and

     simple

    .

     Tell

     me

     more

     about

     yourself

    .

     I

     think

     you

    'll

     find

     it

     interesting

    .

     
    


    I

    'm

     a

    /an

     [

    Name

     of

     character

    ]

     from

     [

    Name

     of

     fictional

     universe

    ]

     
    


    I

    'm

     in

     my

     [

    age

    ]

     year

     old

     when

     I

     joined

     this

     fictional

     universe

    .

     I

    'm

     [

    occupation

    ].

     
    


    I

    'm

     from

     [

    Your

     hometown

     or

     place

     of

     birth

    ].

     I

    'm

     a

    /an

     [

    language

    ]

     native

     speaker

    .
    


    I

    've

     always

     been

     fascinated

     by

     [

    character

    's

     favorite

     hobby

     or

     interest

    ].

     And

     I

     love

     [

    character

    's

     favorite

     topic

    ].

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Paris

     is

     the

     largest

     city

     and

     the

     seat

     of

     the

     French

     government

    ,

     capital

     of

     the

     French

     Departments

    ,

     the

     capital

     of

     the

     French

     region

     Haut

    s

    -de

    -F

    rance

    ,

     and

     a

     major

     industrial

    ,

     cultural

    ,

     and

     tourist

     center

    .

     It

     is

     home

     to

     the

     Lou

    vre

    ,

     E

    iff

    el

     Tower

    ,

     the

     Notre

    -D

    ame

     Cathedral

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Arc

     de

     Tri

    omp

    he

    .

     Paris

     has

     a

     rich

     history

    ,

     including

     the

     origins

     of

     Roman

    ,

     Gothic

    ,

     Bar

    oque

    ,

     and

     Modern

     architectural

     styles

    .

     The

     city

     is

     also

     home

     to

     some

     of

     France

    's

     most

     famous

     museums

    ,

     including

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     the

     Mus

    ée

     Rod

    in

    ,

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     not

     as

     clear

    -cut

     as

     it

     may

     seem

    .

     However

    ,

     based

     on

     current

     trends

    ,

     there

     are

     several

     areas

     where

     we

     can

     expect

     to

     see

     significant

     advancements

     in

     AI

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

     healthcare

     to

     analyze

     patient

     data

    ,

     diagnose

     diseases

    ,

     and

     develop

     personalized

     treatments

    .

     In

     the

     future

    ,

     we

     can

     expect

     to

     see

     even

     more

     AI

    -driven

     healthcare

     innovations

    ,

     such

     as

     more

     accurate

     disease

     diagnosis

    ,

     more

     efficient

     drug

     development

    ,

     and

     more

     personalized

     patient

     care

    .
    


    2

    .

     Increased

     use

     of

     AI

     in

     transportation

    :

     AI

     is

     already

     being

     used

     in

     transportation

     to

     improve

     safety

    ,

     reduce

     traffic

     congestion

    ,

     and

     optimize

     routes

    .

     In

     the

     future

    ,

     we

     can

    



```python
llm.shutdown()
```
