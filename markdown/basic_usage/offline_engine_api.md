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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.38it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.37it/s]


    2026-04-15 08:52:59,399 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 08:52:59] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:38,  2.77s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:38,  2.77s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:38,  2.77s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:38,  2.77s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:38,  2.77s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:23,  2.29it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.66it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.66it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:03<00:06,  6.66it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:03<00:06,  6.66it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:03<00:06,  6.66it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:03<00:06,  6.66it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:03<00:06,  6.66it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:03<00:06,  6.66it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.66it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.66it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.60it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.60it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.60it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.60it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.60it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.60it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.60it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.60it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.60it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.60it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 21.58it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 21.58it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 21.58it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 21.58it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 21.58it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 21.58it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 21.58it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 21.58it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 21.58it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 21.58it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.34it/s]

    Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.34it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.34it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.10it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.10it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.10it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.10it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.10it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.10it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.10it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.10it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.10it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.10it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:03<00:00, 48.03it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:03<00:00, 48.03it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.39it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=114.90 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=114.87 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=114.87 GB):   3%|▎         | 2/58 [00:00<00:03, 18.63it/s]Capturing num tokens (num_tokens=7168 avail_mem=114.87 GB):   3%|▎         | 2/58 [00:00<00:03, 18.63it/s]Capturing num tokens (num_tokens=6656 avail_mem=114.86 GB):   3%|▎         | 2/58 [00:00<00:03, 18.63it/s]Capturing num tokens (num_tokens=6144 avail_mem=114.84 GB):   3%|▎         | 2/58 [00:00<00:03, 18.63it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=114.84 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=114.83 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=114.84 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=114.83 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=114.83 GB):   9%|▊         | 5/58 [00:00<00:02, 21.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=114.83 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.74it/s]Capturing num tokens (num_tokens=3840 avail_mem=114.83 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.74it/s]Capturing num tokens (num_tokens=3584 avail_mem=114.82 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.74it/s]Capturing num tokens (num_tokens=3328 avail_mem=114.82 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.74it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=114.82 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.74it/s]Capturing num tokens (num_tokens=3072 avail_mem=114.82 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.52it/s]Capturing num tokens (num_tokens=2816 avail_mem=114.82 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.52it/s]Capturing num tokens (num_tokens=2560 avail_mem=114.81 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.52it/s]Capturing num tokens (num_tokens=2304 avail_mem=114.81 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.52it/s]Capturing num tokens (num_tokens=2048 avail_mem=114.80 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=114.80 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.52it/s]Capturing num tokens (num_tokens=1792 avail_mem=114.80 GB):  31%|███       | 18/58 [00:00<00:01, 34.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=114.77 GB):  31%|███       | 18/58 [00:00<00:01, 34.59it/s]Capturing num tokens (num_tokens=1280 avail_mem=114.76 GB):  31%|███       | 18/58 [00:00<00:01, 34.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=114.74 GB):  31%|███       | 18/58 [00:00<00:01, 34.59it/s]

    Capturing num tokens (num_tokens=960 avail_mem=114.76 GB):  31%|███       | 18/58 [00:00<00:01, 34.59it/s] Capturing num tokens (num_tokens=896 avail_mem=114.75 GB):  31%|███       | 18/58 [00:00<00:01, 34.59it/s]Capturing num tokens (num_tokens=896 avail_mem=114.75 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=832 avail_mem=114.75 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=768 avail_mem=114.75 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=704 avail_mem=114.74 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=640 avail_mem=114.74 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=576 avail_mem=114.74 GB):  40%|███▉      | 23/58 [00:00<00:00, 37.23it/s]Capturing num tokens (num_tokens=576 avail_mem=114.74 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=512 avail_mem=114.73 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=480 avail_mem=114.74 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.91it/s]

    Capturing num tokens (num_tokens=448 avail_mem=114.74 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=416 avail_mem=114.74 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=384 avail_mem=114.74 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.91it/s]Capturing num tokens (num_tokens=384 avail_mem=114.74 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=352 avail_mem=114.73 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=320 avail_mem=114.73 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=288 avail_mem=114.72 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.26it/s]Capturing num tokens (num_tokens=256 avail_mem=114.72 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=240 avail_mem=114.72 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.26it/s]Capturing num tokens (num_tokens=240 avail_mem=114.72 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.97it/s]Capturing num tokens (num_tokens=224 avail_mem=114.72 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.97it/s]

    Capturing num tokens (num_tokens=208 avail_mem=114.71 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.97it/s]Capturing num tokens (num_tokens=192 avail_mem=114.71 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.97it/s]Capturing num tokens (num_tokens=176 avail_mem=114.71 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.97it/s]Capturing num tokens (num_tokens=160 avail_mem=114.71 GB):  66%|██████▌   | 38/58 [00:01<00:00, 40.97it/s]Capturing num tokens (num_tokens=160 avail_mem=114.71 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=144 avail_mem=114.70 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=128 avail_mem=114.70 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=112 avail_mem=114.70 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.27it/s]Capturing num tokens (num_tokens=96 avail_mem=114.69 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.27it/s] Capturing num tokens (num_tokens=80 avail_mem=114.69 GB):  74%|███████▍  | 43/58 [00:01<00:00, 41.27it/s]

    Capturing num tokens (num_tokens=80 avail_mem=114.69 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=64 avail_mem=114.69 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=48 avail_mem=114.68 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=32 avail_mem=114.68 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=28 avail_mem=114.67 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=24 avail_mem=114.67 GB):  83%|████████▎ | 48/58 [00:01<00:00, 41.63it/s]Capturing num tokens (num_tokens=24 avail_mem=114.67 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.34it/s]Capturing num tokens (num_tokens=20 avail_mem=114.67 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.34it/s]Capturing num tokens (num_tokens=16 avail_mem=114.67 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.34it/s]Capturing num tokens (num_tokens=12 avail_mem=114.66 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.34it/s]Capturing num tokens (num_tokens=8 avail_mem=114.66 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.34it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=114.66 GB):  91%|█████████▏| 53/58 [00:01<00:00, 41.34it/s]Capturing num tokens (num_tokens=4 avail_mem=114.66 GB): 100%|██████████| 58/58 [00:01<00:00, 41.72it/s]Capturing num tokens (num_tokens=4 avail_mem=114.66 GB): 100%|██████████| 58/58 [00:01<00:00, 37.70it/s]


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
    Generated text:  Kristin. I'm a PhD candidate in Biochemistry at the University of California, Santa Cruz. I specialize in RNAi and regulatory circuits. RNAi is a biochemical process that allows for short-term gene silencing. My research is broadly focused on finding how RNAi operates at the molecular level, and how it could be used in drug discovery and development. I am also interested in gene regulation in plants and bacteria, and I have a particular interest in how plants and bacteria regulate expression of their own genes.
    My lab is interested in RNAi and how it operates in the context of gene regulation. We are especially interested in the dynamics of
    ===============================
    Prompt: The president of the United States is
    Generated text:  a(n) _____. He or she is the leader of the country and is responsible for the day-to-day operations of the government.
    A. Chief Executive
    B. Chief Minister
    C. Governor
    D. President
    
    A. Chief Executive
    
    The president of the United States is typically referred to as the "Chief Executive" or "Executive Officer" of the country. This title is used in most countries to describe the top leader of a government. While the other options mentioned in the question are not entirely correct, they are not directly related to the role of the president in the United States. The president is responsible for the day-to
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. A. Paris B. London C. New York D. Sydney
    
    The capital of France is Paris. 
    
    Therefore, the correct answer is A. Paris. 
    
    Note: "Sydney" is not the capital city of France. Sydney is the capital of Australia. However, Sydney is considered one of the three cities of the "Big Three" of Australia, along with Melbourne and Canberra. 
    
    The other options are incorrect because London (option B), New York (option C), and Sydney (option D) are not capital cities of France. The capital of France is actually Paris. However, since the question asks for the
    ===============================
    Prompt: The future of AI is
    Generated text:  a new era of open data. It will make the world more inclusive, and it will encourage innovation and collaboration. With the use of AI, we can collect and analyze vast amounts of data, which can then be used to improve our understanding of the world around us. This data can be used to predict trends, analyze customer behavior, and even identify potential risks. By using AI, we can create better tools for decision-making and improve the quality of life for everyone. As the world continues to grow, the future of AI looks promising. AI will continue to advance and evolve, and we can expect to see even more exciting developments in the


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [occupation]. I have a [number] degree in [field of study]. I'm a [job title] at [company name]. I'm passionate about [reason for passion]. I'm always looking for new challenges and opportunities to grow and learn. What do you like to do for fun? I enjoy [activity or hobby]. I'm always looking for new experiences and adventures. What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a cultural and historical center with a rich history dating back to the Middle Ages. Paris is a popular tourist destination and a major economic hub, with a diverse population of around 2.3 million people. The city is home to many renowned museums, including the Musée d'Orsay and the Musée d'Orsay. Paris is also known for its cuisine, with a wide variety of dishes and restaurants serving up to 100 different types of food. The city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased automation: AI is expected to become more integrated into various industries, leading to increased automation of tasks and processes. This could result in the creation of new jobs and the displacement of existing ones.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there will be a need to address privacy and security concerns. This could lead to the development of new technologies and protocols to protect user data and prevent cyber attacks.
    
    3. Enhanced human-computer interaction: AI is likely to become more integrated into human-computer interaction, allowing for more seamless and intuitive
    


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
    Generated text:  [Name], and I'm a/an [Age] year old [Occupation/Profession] with a background in [What is your background in?] and a passion for [Why does this interest you?]. I've always been curious about the world and what makes people tick, and I believe that every person is unique and special, even if they don't know it yet. Whether you're interested in [What do you enjoy doing?], [What are some hobbies/activities you enjoy?], or [What’s something you’re passionate about?], you’re in luck! I’m excited to learn more about you and help you
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the most populous city in Europe and is known for its rich history, beautiful architecture, and famous landmarks such as the Eiffel Tower. It is also the seat of the French government and is home to numerous museums, galleries, and cultural institutions. The city is known for its vibrant nightlife, food scene, and fashion industry, and is a major tourist destination. Paris is a global capital of culture and entertainment and is recognized as one of the world's most important cities. The French capital is also home to many universities, research institutions, and professional organizations. Paris is an important economic center and a leader in the arts
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a number of distinct trends, including:
    
    1. Advancements in machine learning: As the technology for machine learning continues to improve, we can expect to see greater and more sophisticated AI systems emerge. This could lead to even more advanced algorithms and models that can learn from large amounts of data and make more accurate predictions and decisions.
    
    2. Increased use of AI in natural language processing: AI is already playing a major role in natural language processing, but we can expect to see even greater use of this technology in the future. This could involve more sophisticated natural language understanding and generation, as well as more personalized and context-aware


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

    ]

     and

     I

    'm

     [

    Age

    ].

     I

    'm

     a

     [

    Occup

    ation

    ]

     who

     has

     been

     around

     for

     [

    Years

    ]

     and

     I

    've

     always

     been

     [

    X

    ]

     to

     the

     community

    .

     I

     enjoy

     [

    X

    ]

     and

     [

    X

    ],

     and

     I

    'm

     always

     [

    X

    ]

     to

     the

     people

     I

     work

     with

    .

     I

     believe

     that

     [

    X

    ]

     is

     the

     key

     to

     success

     and

     I

    'm

     constantly

     striving

     to

     grow

     and

     learn

    .

     I

    'm

     passionate

     about

     [

    X

    ]

     and

     I

    'm

     looking

     forward

     to

     [

    X

    ]

     with

     you

    .

     What

    's

     your

     name

    ?

     How

     old

     are

     you

    ?

     How

     long

     have

     you

     been

     in

     this

     field

    ?

     What do

     you

     enjoy

     about

     it

    ?

     What

     are

     your

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     south

     of

     the

     country

    ,

     and

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     a

     major

     city

     for

     its

     French

     culture

     and

     cuisine

    ,

     with

     a

     diverse

     range

     of

     shops

    ,

     restaurants

    ,

     and

     entertainment

     options

    .

     It

     has

     a

     vibrant

     nightlife

     and

     a

     long

     history

     of

     revolution

    ,

     revolution

    ,

     revolution

    ,

     dating

     back

     to

     the

     time

     of

     the

     French

     Revolution

    .

     As

     of

     

    2

    0

    2

    1

    ,

     Paris

     has

     a

     population

     of

     over

     

    1

    .

     

    3

     million

     people

    .

     The

     city

     is

     home

     to

     numerous

     museums

    ,

     art

     galleries

    ,

     and

     historical

     sites

    ,

     making

     it

     a

    
    
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

     natural

     language

     processing

    ,

     computer

     vision

    ,

     and

     autonomous

     driving

    .

     AI

     will

     also

     become

     more

     integrated

     into

     human

     decision

    -making

     processes

    ,

     with

     more

     complex

     models

     and

     algorithms

     designed

     to

     take

     into

     account

     a

     wider

     range

     of

     variables

    .
    


    In

     terms

     of

     technological

     advancements

    ,

     we

     can

     expect

     the

     emergence

     of

     new

     forms

     of

     AI

     such

     as

     quantum

     AI

     and

     neural

     networks

    ,

     which

     will

     enable

     machines

     to

     process

     and

     analyze

     data

     in

     ways

     that

     would

     be

     impossible

     with

     traditional

     AI

     systems

    .

     We

     may

     also

     see

     the

     development

     of

     AI

     that

     can

     learn

     and

     adapt

     to

     new

     situations

    ,

     making

     it

     more

     flexible

     and

     adaptable

     to

     different

     tasks

    .
    


    Moreover

    ,

     AI

     will

     continue

     to

     play

     an

    



```python
llm.shutdown()
```
