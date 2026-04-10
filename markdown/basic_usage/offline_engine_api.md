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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.34it/s]


    2026-04-10 16:32:48,886 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-10 16:32:48] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:20,  2.46s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:20,  2.46s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:20,  2.46s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:20,  2.46s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:26,  2.03it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:26,  2.03it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:26,  2.03it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:26,  2.03it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:26,  2.03it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:26,  2.03it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:26,  2.03it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:07,  6.18it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:07,  6.18it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:07,  6.18it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:07,  6.18it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:07,  6.18it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:07,  6.18it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:07,  6.18it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:07,  6.18it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:07,  6.18it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:07,  6.18it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.83it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.83it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.83it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.83it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.83it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:02<00:02, 13.83it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:02<00:02, 13.83it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:02<00:02, 13.83it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:02<00:02, 13.83it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:02<00:01, 21.31it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:02<00:01, 21.31it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:02<00:01, 21.31it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:02<00:01, 21.31it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:02<00:01, 21.31it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:02<00:01, 21.31it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:02<00:01, 21.31it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:02<00:01, 21.31it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:02<00:01, 21.31it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 29.32it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 29.32it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 29.32it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 29.32it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 29.32it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 29.32it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 29.32it/s]

    Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 29.32it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 35.94it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 35.94it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 35.94it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 35.94it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 35.94it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 35.94it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 35.94it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 35.94it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 42.06it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 42.06it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 42.06it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 42.06it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 42.06it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 42.06it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 42.06it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 42.06it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 42.06it/s] 

    Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 42.06it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 17.56it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.22 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.19 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:02, 19.91it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:02, 19.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:02, 19.91it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.19 GB):   3%|▎         | 2/58 [00:00<00:02, 19.91it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.18 GB):   9%|▊         | 5/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.19 GB):   9%|▊         | 5/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.18 GB):   9%|▊         | 5/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.18 GB):   9%|▊         | 5/58 [00:00<00:02, 23.19it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.21it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.18 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.21it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.21it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.21it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.21it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.32it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.16 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.32it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.15 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.32it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.14 GB):  24%|██▍       | 14/58 [00:00<00:01, 34.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.14 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.12 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=960 avail_mem=74.14 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.64it/s] Capturing num tokens (num_tokens=896 avail_mem=74.14 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=832 avail_mem=74.13 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.64it/s]

    Capturing num tokens (num_tokens=768 avail_mem=74.13 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=704 avail_mem=74.13 GB):  34%|███▍      | 20/58 [00:00<00:00, 40.64it/s]Capturing num tokens (num_tokens=704 avail_mem=74.13 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.91it/s]Capturing num tokens (num_tokens=640 avail_mem=74.12 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.91it/s]Capturing num tokens (num_tokens=576 avail_mem=74.12 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.91it/s]Capturing num tokens (num_tokens=512 avail_mem=74.11 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.91it/s]Capturing num tokens (num_tokens=480 avail_mem=74.13 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.91it/s]Capturing num tokens (num_tokens=448 avail_mem=74.12 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.91it/s]Capturing num tokens (num_tokens=416 avail_mem=74.12 GB):  45%|████▍     | 26/58 [00:00<00:00, 44.91it/s]Capturing num tokens (num_tokens=416 avail_mem=74.12 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.77it/s]Capturing num tokens (num_tokens=384 avail_mem=74.12 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.77it/s]Capturing num tokens (num_tokens=352 avail_mem=74.11 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.77it/s]Capturing num tokens (num_tokens=320 avail_mem=74.11 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.77it/s]

    Capturing num tokens (num_tokens=288 avail_mem=74.11 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.77it/s]Capturing num tokens (num_tokens=256 avail_mem=74.10 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.77it/s]Capturing num tokens (num_tokens=240 avail_mem=74.10 GB):  55%|█████▌    | 32/58 [00:00<00:00, 47.77it/s]Capturing num tokens (num_tokens=240 avail_mem=74.10 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.45it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.45it/s]Capturing num tokens (num_tokens=208 avail_mem=74.09 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.45it/s]Capturing num tokens (num_tokens=192 avail_mem=74.09 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.45it/s]Capturing num tokens (num_tokens=176 avail_mem=74.09 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.45it/s]Capturing num tokens (num_tokens=160 avail_mem=74.09 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.45it/s]Capturing num tokens (num_tokens=144 avail_mem=74.08 GB):  66%|██████▌   | 38/58 [00:00<00:00, 49.45it/s]Capturing num tokens (num_tokens=144 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.89it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.89it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.89it/s]

    Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.89it/s] Capturing num tokens (num_tokens=80 avail_mem=74.07 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.89it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.89it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  76%|███████▌  | 44/58 [00:01<00:00, 50.89it/s]Capturing num tokens (num_tokens=48 avail_mem=74.07 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.43it/s]Capturing num tokens (num_tokens=32 avail_mem=74.06 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.43it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.43it/s]Capturing num tokens (num_tokens=24 avail_mem=74.05 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.43it/s]Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.43it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.43it/s]

    Capturing num tokens (num_tokens=12 avail_mem=74.04 GB):  86%|████████▌ | 50/58 [00:01<00:00, 51.43it/s]Capturing num tokens (num_tokens=12 avail_mem=74.04 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=8 avail_mem=74.04 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.55it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  97%|█████████▋| 56/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 40.30it/s]


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
    Generated text:  Anna. I am twelve years old. I like singing and dancing. I am a leader of the club. I always sing and dance in the club. I want to be a leader when I grow up. After school, I often help my friends with their homework. Then I often play basketball with my friends. I often eat ice-cream. I like my life very much. I'm a good boy, and I hope to be a good leader when I grow up. How does Anna feel about her life?
    
    Answer the following questions based on the information above.
    
    1. She likes to sing and dance.
    
    2. She is a
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person in the government of the country. The president is the head of the executive branch, which includes the executive, judicial and legislative branches. The executive branch has the power to make laws and appointments to the executive branch. The president is the head of the executive branch and is elected for a term of four years.
    
    The president is responsible for making decisions that are made by the executive branch. These decisions can include things like:
    - The hiring and firing of federal employees and employees of the federal government
    - The appointment of judges to the Supreme Court of the United States
    - The appointment of federal judges to federal district courts
    
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    
    Does it follow that "Paris is the capital of France, "?
    OPTIONS:
    [a]. yes;
    [b]. it is not possible to tell;
    [c]. no;
    
    [a]. yes;
    
    The statement "Paris is the capital of France" is correct. Therefore, it follows that "Paris is the capital of France, " as the capital of France is defined as the capital city of France, which includes Paris. "
    ===============================
    Prompt: The future of AI is
    Generated text:  largely dependent on the market. Once a sector, it has a long way to go to reach its full potential. The market is not only a major driver of innovation but it also has a role to play in the development of the technology. Thus, as a core player, it must be able to keep pace with the market trends and competitive landscape. As an AI technology provider, we have found the recent developments in the AI space to be quite interesting. A number of high-profile players are opening up their technologies to be used in a wide range of applications. That includes the digital transformation of healthcare, where AI is being used to improve the


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


    Generated text:  [Name], and I'm a [Job Title] at [Company Name]. I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name], and I'm excited to meet you and learn more about your interests and experiences. What can you tell me about yourself? [Name] is a [job title] at [company name
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for business, finance, and entertainment. The city is known for its annual fashion and food festivals, as well as its annual Eiffel Tower celebration. Paris is a popular tourist destination and a cultural hub for the French people. It is a major economic and political center in Europe. The city is also home to
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies are expected to continue to improve and become more integrated into our daily lives, from self-driving cars and robots to personalized medicine and virtual assistants. Additionally, AI will likely continue to be used for a wide range of applications, from improving healthcare outcomes to enhancing customer service and improving the efficiency of businesses. However, there are also potential risks and challenges associated with the development and deployment of AI, including concerns about bias, privacy, and security. As we continue to see the development of new AI technologies, it is likely that we
    


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
    Generated text:  __________ and I'm a(n) __________. I love __________ because ___________________. What's your name? What's your profession? How long have you been working in this field? How many years have you been in this job? What do you do? What do you enjoy? What do you like to do in your free time? What do you want to learn? What's your dream job? How do you like to spend your weekends? What are your hobbies? What do you do for a living? What are your hobbies? What do you do for a living? What are your hobbies? What do you do for a
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the second-largest city in the European Union and is known for its iconic Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. 
    
    - Paris, officially known as the Île-de-France department in France, has a population of over 16 million people.
    - The city has been an important center of European and global culture since its founding in the 8th century.
    - Its history dates back to the construction of the first Roman fort in the region.
    - Paris has been a UNESCO World Heritage Site since 1992. 
    - It is home to numerous museums, including the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve a wide range of developments and advancements, including:
    
    1. Increased capabilities: AI is likely to become even more capable, with more complex algorithms, faster processing power, and more sophisticated machine learning models.
    
    2. Autonomous vehicles: Self-driving cars, drones, and other autonomous vehicles are likely to become more widespread, offering a safer and more efficient alternative to human drivers.
    
    3. Increased use of AI for security and surveillance: AI-powered security systems and surveillance technologies are likely to become more advanced, providing better protection against cyber threats and other security challenges.
    
    4. AI in healthcare: AI is likely to be used in medicine, with


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

     Emily

    .

     I

    'm

     a

     skilled

     web

     developer

     and

     have

     worked

     on

     numerous

     websites

    ,

     including

     a

     popular

     e

    -commerce

     site

    .

     I

    've

     been

     passionate

     about

     technology

     and

     web

     development

     for

     as

     long

     as

     I

     can

     remember

    .

     I

     enjoy

     creating

     beautiful

     and

     functional

     websites

     that

     are

     easy

     to

     navigate

     and

     visually

     appealing

    .

     I

     also

     have

     a

     keen

     eye

     for

     design

    ,

     and

     I

     strive

     to

     always

     make

     my

     websites

     stand

     out

     from

     the

     crowd

    .

     I

    'm

     very

     dedicated

     to

     staying

     up

    -to

    -date

     with

     the

     latest

     technology

     and

     web

     development

     trends

     and

     always

     seek

     out

     new

     opportunities

     to

     learn

     and

     improve

    .

     I

    'm

     also

     a

     good

     communicator

     and

     always

     enjoy

     helping

     others

     with

     their

     online

     projects

    .

     So

    ,

     if

     you

    're

     looking

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     the

     largest

     city

     in

     the

     country

     and

     also

     the

     capital

     of

     France

    .


    Key

     facts

     about

     Paris

    :


    -

     It

     is

     the

     largest

     city

     in

     France

    


    -

     It

     is

     known

     for

     its

     historical

     landmarks

    ,

     including

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

    


    -

     It

     is

     a

     cultural

     and

     financial

     center

    ,

     known

     for

     its

     fashion

     industry

     and

     arts

     scene

    


    -

     It

     is

     located

     in

     the

     south

     of

     France

     near

     the

     Mediterranean

     Sea

    


    -

     It

     has

     a

     rich

     history

     dating

     back

     to

     the

     medieval

     period

    


    -

     It

     is

     home

     to

     many

     museums

    ,

     including

     the

     Lou

    vre

    ,

     the

     Mus

    ée

     d

    '

    Or

    say

    ,

     and

     the

     Mus

    ée

     de

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     rapidly

     changing

    ,

     with

     new

     developments

     and

     applications

     emerging

     all

     the

     time

    .

     Here

     are

     some

     potential

     trends

     that

     could

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Automation

     of

     tasks

    :

     Automation

     is

     likely

     to

     become

     more

     prevalent

     in

     AI

     applications

    ,

     with

     AI

     systems

     becoming

     more

     efficient

     at

     performing

     tasks

     that

     are

     typically

     done

     by

     humans

    .

     This

     could

     lead

     to

     the

     widespread

     adoption

     of

     AI

     in

     a

     wide

     range

     of

     industries

    ,

     from

     manufacturing

     to

     healthcare

     to

     finance

    .
    


    2

    .

     Aug

    mented

     AI

    :

     As

     AI

     systems

     become

     more

     advanced

    ,

     they

     may

     start

     to

     produce

     more

     complex

     and

     intuitive

     AI

     that

     can

     perform

     tasks

     more

     effectively

     than

     humans

    .

     This

     could

     lead

     to

     more

     personalized

     and

     intuitive

     AI

    



```python
llm.shutdown()
```
