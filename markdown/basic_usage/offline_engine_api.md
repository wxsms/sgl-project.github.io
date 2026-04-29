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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.11it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.10it/s]


    2026-04-29 05:57:52,025 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-29 05:57:52] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.39it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.15it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:11,  4.15it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.33it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.33it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.09it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.09it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.09it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.09it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.09it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.09it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.09it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.09it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.09it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.09it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.09it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:05<00:01, 15.09it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:05<00:00, 23.74it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:05<00:00, 23.74it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:05<00:00, 23.74it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:05<00:00, 23.74it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:05<00:00, 23.74it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:05<00:00, 23.74it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:05<00:00, 23.74it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:05<00:00, 23.74it/s]

    Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:05<00:00, 23.74it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:05<00:00, 23.74it/s]Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:05<00:00, 23.74it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 32.53it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 32.53it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 32.53it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 32.53it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 32.53it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 32.53it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 32.53it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.81it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.86 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.83 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.83 GB):   3%|▎         | 2/58 [00:00<00:03, 18.11it/s]Capturing num tokens (num_tokens=7168 avail_mem=70.82 GB):   3%|▎         | 2/58 [00:00<00:03, 18.11it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.82 GB):   3%|▎         | 2/58 [00:00<00:03, 18.11it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.82 GB):   3%|▎         | 2/58 [00:00<00:03, 18.11it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=70.82 GB):   9%|▊         | 5/58 [00:00<00:02, 21.03it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.82 GB):   9%|▊         | 5/58 [00:00<00:02, 21.03it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.81 GB):   9%|▊         | 5/58 [00:00<00:02, 21.03it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.80 GB):   9%|▊         | 5/58 [00:00<00:02, 21.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.80 GB):   9%|▊         | 5/58 [00:00<00:02, 21.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=70.80 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.80 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.79 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.79 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.79 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=70.79 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.37it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.79 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.78 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.78 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.77 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.77 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.77 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.38it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.77 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.77 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.75 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s]Capturing num tokens (num_tokens=960 avail_mem=70.76 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s] Capturing num tokens (num_tokens=896 avail_mem=70.76 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s]

    Capturing num tokens (num_tokens=832 avail_mem=70.75 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s]Capturing num tokens (num_tokens=768 avail_mem=70.75 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.78it/s]Capturing num tokens (num_tokens=768 avail_mem=70.75 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.34it/s]Capturing num tokens (num_tokens=704 avail_mem=70.75 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.34it/s]Capturing num tokens (num_tokens=640 avail_mem=70.74 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.34it/s]Capturing num tokens (num_tokens=576 avail_mem=70.74 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.34it/s]Capturing num tokens (num_tokens=512 avail_mem=70.73 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.34it/s]Capturing num tokens (num_tokens=480 avail_mem=70.74 GB):  43%|████▎     | 25/58 [00:00<00:00, 42.34it/s]Capturing num tokens (num_tokens=480 avail_mem=70.74 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=448 avail_mem=70.74 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=416 avail_mem=70.74 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=384 avail_mem=70.74 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=352 avail_mem=70.73 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.57it/s]

    Capturing num tokens (num_tokens=320 avail_mem=70.73 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=288 avail_mem=70.73 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.57it/s]Capturing num tokens (num_tokens=288 avail_mem=70.73 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.63it/s]Capturing num tokens (num_tokens=256 avail_mem=70.72 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.63it/s]Capturing num tokens (num_tokens=240 avail_mem=70.72 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.63it/s]Capturing num tokens (num_tokens=224 avail_mem=70.72 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.63it/s]Capturing num tokens (num_tokens=208 avail_mem=70.71 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.63it/s]Capturing num tokens (num_tokens=192 avail_mem=70.71 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.63it/s]Capturing num tokens (num_tokens=176 avail_mem=70.71 GB):  62%|██████▏   | 36/58 [00:01<00:00, 46.63it/s]Capturing num tokens (num_tokens=176 avail_mem=70.71 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.08it/s]Capturing num tokens (num_tokens=160 avail_mem=70.71 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.08it/s]Capturing num tokens (num_tokens=144 avail_mem=70.70 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.08it/s]Capturing num tokens (num_tokens=128 avail_mem=70.70 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.08it/s]

    Capturing num tokens (num_tokens=112 avail_mem=70.70 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.08it/s]Capturing num tokens (num_tokens=96 avail_mem=70.69 GB):  72%|███████▏  | 42/58 [00:01<00:00, 48.08it/s] Capturing num tokens (num_tokens=96 avail_mem=70.69 GB):  81%|████████  | 47/58 [00:01<00:00, 48.43it/s]Capturing num tokens (num_tokens=80 avail_mem=70.69 GB):  81%|████████  | 47/58 [00:01<00:00, 48.43it/s]Capturing num tokens (num_tokens=64 avail_mem=70.69 GB):  81%|████████  | 47/58 [00:01<00:00, 48.43it/s]Capturing num tokens (num_tokens=48 avail_mem=70.68 GB):  81%|████████  | 47/58 [00:01<00:00, 48.43it/s]Capturing num tokens (num_tokens=32 avail_mem=70.68 GB):  81%|████████  | 47/58 [00:01<00:00, 48.43it/s]Capturing num tokens (num_tokens=28 avail_mem=70.67 GB):  81%|████████  | 47/58 [00:01<00:00, 48.43it/s]Capturing num tokens (num_tokens=28 avail_mem=70.67 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.72it/s]Capturing num tokens (num_tokens=24 avail_mem=70.67 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.72it/s]Capturing num tokens (num_tokens=20 avail_mem=70.67 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.72it/s]Capturing num tokens (num_tokens=16 avail_mem=70.67 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.72it/s]

    Capturing num tokens (num_tokens=12 avail_mem=70.66 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.72it/s]Capturing num tokens (num_tokens=8 avail_mem=70.66 GB):  90%|████████▉ | 52/58 [00:01<00:00, 48.72it/s] Capturing num tokens (num_tokens=8 avail_mem=70.66 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.87it/s]Capturing num tokens (num_tokens=4 avail_mem=70.66 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.87it/s]Capturing num tokens (num_tokens=4 avail_mem=70.66 GB): 100%|██████████| 58/58 [00:01<00:00, 42.54it/s]


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
    Generated text:  Kanya, and I'm an English student studying in a college. I'm here to find out how to use social media effectively for a career.
    
    What are some practical tips for effective social media usage in a job context? Are there any particular social media platforms that are more suitable for certain types of jobs, or should I explore all of them? What are some ways to stay updated on the latest trends and job opportunities related to social media? Finally, how can I develop my own social media presence to make a strong impact on my career? Please provide examples of your own experiences, and also point out any common pitfalls and the best ways
    ===============================
    Prompt: The president of the United States is
    Generated text:  5 feet 4 inches tall. Convert the president's height into centimeters. (Note: 1 foot = 30.5 cm)
    
    To convert the president's height from feet and inches to centimeters, we need to follow these steps:
    
    1. Convert the feet to centimeters.
    2. Convert the inches to centimeters.
    3. Add the two results together to get the total height in centimeters.
    
    First, let's convert the feet to centimeters. Since there are 30.5 cm in a foot, we multiply the number of feet by 30.5 cm:
    \[ 5 \
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Brussels
    C. Lyon
    D. Nice
    Answer:
    A
    
    What type of expression is the following? 3/5+2/5=1
    A. True expression
    B. False expression
    C. Unable to determine
    Answer:
    B
    
    Which of the following statements is true?
    A. The expression 3x^2 - 2x + 1 cannot be factored further.
    B. The expression 3x^2 - 2x + 1 can be factored further.
    C. The expression 3x^2 - 2x + 
    ===============================
    Prompt: The future of AI is
    Generated text:  changing rapidly, with breakthroughs that could transform the way we work, learn, and interact with technology. As AI continues to evolve and deepen in sophistication, it will likely lead to new innovations and applications that have the potential to revolutionize industries and enhance quality of life for individuals and societies. However, it is important to address the ethical concerns and potential risks associated with the development and use of AI. For example, AI systems that are designed to perpetuate discrimination or bias could have significant consequences for people and societies, and it is crucial to ensure that AI is developed and used in a way that is fair, transparent, and aligned with societal


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is the largest city in France and the second-largest city in the European Union. Paris is known for its rich history, beautiful architecture, and vibrant culture. It is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its annual festivals and events, including the Eiffel Tower Parade and the World Cup of Lights. The city is a popular tourist destination and a cultural hub for France and the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations and the responsible use of AI. This will include issues such as bias, transparency, accountability, and the potential for AI to be used for harmful purposes.
    
    3. Increased focus on privacy and security: As AI becomes more integrated into our daily lives, there will be a greater emphasis on privacy and security. This
    


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
    Generated text:  [Name], and I'm a [Age] year old AI assistant. I'm programmed to be a friendly and helpful virtual assistant to anyone who interacts with me. Let me know if there's anything specific you'd like me to help you with. Welcome, and please feel free to ask me anything you'd like to know! What's your name, and how can I assist you today? [Name] [Age] [AI Assistant] Hello, I'm [Name] and I'm a [Age] year old AI assistant. I'm here to provide you with helpful information and support whenever you need it. How can I assist
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    **Explanation:**
    Paris, the historical capital city of France, is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Montmartre, which are some of the most famous attractions in the city. The city is also famous for its art, culture, and cuisine, attracting millions of tourists every year. However, as of 2023, the population of Paris is approximately 2.1 million. 
    
    **Insert any relevant historical or cultural fact about Paris here, if applicable.** 
    
    **Example:**
    "Paris has been the seat of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly uncertain, but some possible trends that are likely to shape its development include:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is expected to become more integrated with human intelligence. This may lead to new forms of AI that combine the best of both worlds - for example, AI that can understand and respond to the emotions and feelings of its human users.
    
    2. Artificial general intelligence: Artificial general intelligence (AGI) refers to AI that can perform any cognitive task, including learning, problem-solving, and decision-making. While AGI may be possible in the future, it is currently considered a theoretical concept and is


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

    Occup

    ation

    ]

     who

     has

     been

     [

    Number

     of

     Years

     in

     Industry

    ]

     years

     in

     the

     industry

    .

     My

     expertise

     lies

     in

     [

    field

     of

     expertise

    ],

     and

     I

     have

     a

     passion

     for

     [

    occupation

    ].

     I

     am

     [

    Age

    ]

     years

     old

    ,

     [

    Height

    ]

     inches

     tall

    ,

     and

     [

    Weight

    ]

     pounds

    .

     I

     have

     a

     [

    Characteristic

    ]

     personality

    ,

     and

     my

     [

    Strength

    s

    /

    Weak

    ness

    es

    ]

     include

     [

    Example

     Strength

    s

    ]

     and

     [

    Example

     Weak

    ness

    es

    ].

     I

     believe

     that

     [

    Field

     of

     Interest

    /

    Interest

    ]

     will

     bring

     me

     a

     lot

     of

     [

    positive

    /m

    ixed

     negative

    ]

     impact

     to

     the

     industry

     and

     to

     my

     personal

     life

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     known

     as

     the

     "

    City

     of

     Light

    "

     for

     its

     illuminated

     cath

    ed

    r

    als

    ,

     vibrant

     nightlife

    ,

     and

     cultural

     richness

    .

     It

     is

     located

     on

     the

     Se

    ine

     River

     and

     has

     a

     rich

     history

     dating

     back

     to

     Roman

     times

    ,

     with

     attractions

     such

     as

     the

     Lou

    vre

     and

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     known

     for

     its

     fashion

    ,

     cuisine

    ,

     and

     annual

     festivals

     like

     the

     F

    ête

     de

     l

    '

    Autom

    ne

    .

     Paris

     is

     the

     third

    -largest

     city

     in

     Europe

     and

     is

     a

     major

     economic

     and

     cultural

     hub

    .

     It

     is

     the

     oldest

     city

     in

     the

     world

     and

     has

     played

     a

     significant

     role

     in

     European

     history

    .

     Its

     blend

     of

     classical

     and

     modern

     architecture

     and

     its

     vibrant

     urban

     atmosphere

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     very

     different

     from

     its

     current

     state

    .

     Here

     are

     some

     possible

     trends

     in

     the

     field

     of

     AI

    :
    


    1

    .

     Increased

     precision

     and

     accuracy

    :

     As

     AI

     technology

     advances

    ,

     we

     may

     see

     an

     increase

     in

     the

     precision

     and

     accuracy

     of

     its

     predictions

     and

     decisions

    .

     This

     will

     be

     due

     to

     the

     use

     of

     machine

     learning

     algorithms

     that

     can

     learn

     from

     large

     amounts

     of

     data

    ,

     which

     can

     help

     AI

     systems

     become

     more

     accurate

     and

     reliable

    .
    


    2

    .

     Integration

     with

     human

     intelligence

    :

     AI

     is

     likely

     to

     become

     more

     integrated

     with

     human

     intelligence

    ,

     as

     AI

     systems

     learn

     from

     human

     decision

    -making

    .

     This

     will

     enable

     AI

     systems

     to

     make

     more

     informed

     decisions

     that

     consider

     both

     human

     and

     AI

     perspectives

    .
    


    3

    .

     Advanced

    



```python
llm.shutdown()
```
