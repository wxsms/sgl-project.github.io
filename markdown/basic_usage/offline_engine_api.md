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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.37it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.37it/s]


    2026-04-09 17:57:41,820 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-09 17:57:41] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:35,  2.72s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]

    Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.84it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.65it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:03<00:08,  5.65it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:03<00:08,  5.65it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:03<00:08,  5.65it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 11.97it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]

    Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 11.97it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.08it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.08it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.08it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.08it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.08it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.08it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.08it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.08it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.80it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.80it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.80it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.80it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.80it/s]

    Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.80it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.80it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.90it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.90it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.90it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.90it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.90it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.90it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.90it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.38it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.38it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.38it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.38it/s] 

    Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.38it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.38it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.38it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.76it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.76it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=131.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.29it/s]Capturing num tokens (num_tokens=7168 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.29it/s]Capturing num tokens (num_tokens=6656 avail_mem=131.63 GB):   3%|▎         | 2/58 [00:00<00:03, 18.29it/s]Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   3%|▎         | 2/58 [00:00<00:03, 18.29it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=131.64 GB):   9%|▊         | 5/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=5120 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):   9%|▊         | 5/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=131.63 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.11it/s]Capturing num tokens (num_tokens=3840 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.11it/s]Capturing num tokens (num_tokens=3584 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.11it/s]Capturing num tokens (num_tokens=3328 avail_mem=131.62 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.11it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=131.61 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.11it/s]Capturing num tokens (num_tokens=3072 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=2816 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=2560 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=2304 avail_mem=131.61 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=2048 avail_mem=131.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.87it/s]Capturing num tokens (num_tokens=1792 avail_mem=131.60 GB):  31%|███       | 18/58 [00:00<00:01, 35.78it/s]Capturing num tokens (num_tokens=1536 avail_mem=131.60 GB):  31%|███       | 18/58 [00:00<00:01, 35.78it/s]Capturing num tokens (num_tokens=1280 avail_mem=131.59 GB):  31%|███       | 18/58 [00:00<00:01, 35.78it/s]Capturing num tokens (num_tokens=1024 avail_mem=131.57 GB):  31%|███       | 18/58 [00:00<00:01, 35.78it/s]

    Capturing num tokens (num_tokens=960 avail_mem=131.58 GB):  31%|███       | 18/58 [00:00<00:01, 35.78it/s] Capturing num tokens (num_tokens=896 avail_mem=131.58 GB):  31%|███       | 18/58 [00:00<00:01, 35.78it/s]Capturing num tokens (num_tokens=896 avail_mem=131.58 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.91it/s]Capturing num tokens (num_tokens=832 avail_mem=131.58 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.91it/s]Capturing num tokens (num_tokens=768 avail_mem=131.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.91it/s]Capturing num tokens (num_tokens=704 avail_mem=131.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.91it/s]Capturing num tokens (num_tokens=640 avail_mem=131.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.91it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  40%|███▉      | 23/58 [00:00<00:00, 35.91it/s]Capturing num tokens (num_tokens=576 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.53it/s]Capturing num tokens (num_tokens=512 avail_mem=131.56 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.53it/s]

    Capturing num tokens (num_tokens=480 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.53it/s]Capturing num tokens (num_tokens=448 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.53it/s]Capturing num tokens (num_tokens=416 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.53it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  48%|████▊     | 28/58 [00:00<00:00, 38.53it/s]Capturing num tokens (num_tokens=384 avail_mem=131.57 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=352 avail_mem=131.56 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=320 avail_mem=131.56 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=288 avail_mem=131.55 GB):  57%|█████▋    | 33/58 [00:00<00:00, 40.89it/s]Capturing num tokens (num_tokens=256 avail_mem=131.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  57%|█████▋    | 33/58 [00:01<00:00, 40.89it/s]Capturing num tokens (num_tokens=240 avail_mem=131.55 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.38it/s]Capturing num tokens (num_tokens=224 avail_mem=131.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.38it/s]

    Capturing num tokens (num_tokens=208 avail_mem=131.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.38it/s]Capturing num tokens (num_tokens=192 avail_mem=131.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.38it/s]Capturing num tokens (num_tokens=176 avail_mem=131.54 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.38it/s]Capturing num tokens (num_tokens=160 avail_mem=131.53 GB):  66%|██████▌   | 38/58 [00:01<00:00, 42.38it/s]Capturing num tokens (num_tokens=160 avail_mem=131.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=144 avail_mem=131.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=128 avail_mem=131.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=112 avail_mem=131.53 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=96 avail_mem=131.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.62it/s] Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  74%|███████▍  | 43/58 [00:01<00:00, 43.62it/s]Capturing num tokens (num_tokens=80 avail_mem=131.52 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=64 avail_mem=131.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.16it/s]

    Capturing num tokens (num_tokens=48 avail_mem=131.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=32 avail_mem=131.51 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=28 avail_mem=131.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  83%|████████▎ | 48/58 [00:01<00:00, 44.16it/s]Capturing num tokens (num_tokens=24 avail_mem=131.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=20 avail_mem=131.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=16 avail_mem=131.50 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=12 avail_mem=131.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=8 avail_mem=131.49 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.71it/s] Capturing num tokens (num_tokens=4 avail_mem=131.48 GB):  91%|█████████▏| 53/58 [00:01<00:00, 44.71it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 45.53it/s]Capturing num tokens (num_tokens=4 avail_mem=131.48 GB): 100%|██████████| 58/58 [00:01<00:00, 39.26it/s]


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
    Generated text:  Karen and I am a licensed professional counselor specializing in child and adolescent psychology. My speciality is working with adolescents and young adults with anxiety, depression, eating disorders, trauma, grief, and other psychological disorders. I have extensive experience working with children and adolescents, and have helped numerous clients overcome their mental health issues and achieve long-term psychological and emotional stability. My approach is client-centered and individualized, utilizing evidence-based therapies to help clients achieve their goals. I have extensive experience working with children and adolescents, and have helped numerous clients overcome their mental health issues and achieve long-term psychological and emotional stability. My approach is client-centered and individualized
    ===============================
    Prompt: The president of the United States is
    Generated text:  a职位的英文翻译
    The president of the United States is a position.
    
    是的,这个翻译是正确的。"position"的正确英文翻译是"position"。"The president"的正确英文翻译是"the president"。所以整个句子翻译为:"美国的总统是一个职位。" 这个句子翻译准确表达了原句的意思。不过,如果希望更正式一些的表达,可以是:
    
    "美国总统是一个职位。"
    
    这样表达更加正式和正式。所以翻译为:"The President of the United States is a position." 这个句子翻译也准确表达了原句的意思。不过,
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which has a population of about 2.3 million. Paris is located at the base of the Îtarrive mountain range, which is a geologically active area. The mountain range is believed to be about 2,600 years old. Scientists have found that the upper part of the mountain range has experienced some seismic activity during the last 100 years. It is known that the depth of the mountain range is about 10 kilometers.
    
    (a) How much is the depth of the mountain range? (Round to the nearest tenth of a kilometer.)
    
    (b) What is the surface area of the
    ===============================
    Prompt: The future of AI is
    Generated text:  here: the internet of things, which is the Internet of Things, is a system of machines that can exchange data through networks, through sensors and actuators, and through the Internet. This is where we are headed, with many smart homes, smart cities, smart cars, smart health systems, smart manufacturing, etc. But what does the Internet of Things mean for the future of AI? This topic will be discussed in the section of "AI for AI" (Artificial Intelligence for Artificial Intelligence).
    
    ## The Internet of Things and AI
    
    ### The Internet of Things
    
    The Internet of Things is a system of machines that can exchange data through


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can I expect from our conversation? [Name] is a [job title] at [company name]. I'm excited to meet you and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic Eiffel Tower and the annual Eiffel Tower Festival. It is the largest city in France and the third largest in the world, with a population of over 2.7 million people. Paris is a cultural and historical center, known for its rich history, art, and cuisine. It is also a major transportation hub, with many major highways and airports located in the city. The city is home to many world-renowned museums, including the Louvre and the Musée d'Orsay. Paris is also known for its fashion industry, with many famous designers and boutiques located in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends, including:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence, allowing for more complex and nuanced interactions between the two.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to advance, it is likely to be used in even more areas, including diagnosis, treatment, and patient care.
    
    3. Greater use of AI in education: AI is already being used in education to personalize learning experiences, improve student engagement, and
    


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
    Generated text:  [name] and I'm a [occupation] who has been living in [city] for [number of years] years. I am always looking for ways to [describe my hobbies, interests, or goals]. 
    
    I am always eager to learn new things and to share my knowledge with those who need it. I believe in [describe a principle or belief that you hold dear to your heart], and I am committed to [describe how you maintain that belief]. I am always willing to travel, try new foods, and spend time in nature. I'm always looking for new ways to make my life more interesting and enjoyable. 
    
    If
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a cosmopolitan city with a rich cultural heritage and is considered one of the world's most popular tourist destinations. The city is home to over 2 million inhabitants and is known for its museums, theaters, and cafes. Paris is also home to several famous landmarks, such as the Eiffel Tower, Notre Dame Cathedral, and the Louvre Museum. It is the seat of the French government and a major financial hub. Paris is considered one of the most important cities in the world and is home to many important French art, literature, and music institutions. The city has a long and storied history and is home
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by several key trends:
    
    1. Advancements in machine learning: As AI technology continues to advance, we are likely to see more sophisticated algorithms that can learn from large datasets, make predictions, and adapt to new situations.
    
    2. Integration with human intelligence: AI will likely become more integrated with human intelligence, allowing machines to perform tasks that require human-like decision-making and problem-solving.
    
    3. Greater focus on ethical considerations: As AI systems become more complex and responsible, there will be an increased focus on addressing ethical concerns, such as bias, transparency, and accountability.
    
    4. Rise of "soft skills": As AI becomes


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

    ],

     and

     I

     am

     a

     [

    Your

     Profession

    /

    Title

    ]

     with

     over

     [

    Number

     of

     Years

    ]

     years

     of

     experience

     in

     the

     industry

    .

     My

     background

     is

     in

     [

    Industry

    ],

     and

     I

     have

     always

     been

     fascinated

     by

     the

     challenges

     and

     opportunities

     that

     come

     with

     my

     profession

    .

     I

     am

     always

     eager

     to

     learn

     and

     improve

     my

     skills

    ,

     and

     I

     am

     always

     looking

     for

     ways

     to

     make

     a

     positive

     impact

     on

     the

     world

    .

     I

     enjoy

     being

     creative

     and

     innovative

    ,

     and

     I

     believe

     that

     with

     the

     right

     mindset

     and

     approach

    ,

     anyone

     can

     achieve

     their

     goals

    .

     I

     am

     [

    Your

     Inter

    ests

    /

    Inter

    ests

    ],

     and

     I

     am

     always

     looking

     for

     new

     ways

     to

     grow

     and

     learn

    .

     I

     am

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     Se

    ine

     River

     in

     the

     center

     of

     the

     country

    .


    Paris

     is

     the

     largest

     city

     in

     France

     and

     the

     capital

     of

     France

    .

     It

     is

     home

     to

     the

     most

     famous

     landmarks

     in

     Europe

    ,

     including

     the

     E

    iff

    el

     Tower

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     has

     a

     rich

     history

     dating

     back

     to

     ancient

     times

    ,

     with

     its

     origins

     tracing

     back

     to

     the

     Roman

     Empire

    .

     Paris

     is

     known

     for

     its

     diverse

     culture

    ,

     fashion

    ,

     art

    ,

     and

     cuisine

    ,

     and

     has

     hosted

     numerous

     world

     events

    ,

     including

     the

     

    2

    0

    1

    2

     Summer

     Olympics

    .

     Today

    ,

     Paris

     continues

     to

     be

     one

     of

     the

     most

     important

     cities

     in

     Europe

    ,

     known

     for

     its

     vibrant

     nightlife

    ,

     art

     scene

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     looking

     increasingly

     like

     a

     roller

     coaster

    ,

     with

     exciting

     new

     developments

     and

     potential

     downs

    ides

     to

     consider

    .

     Here

     are

     a

     few

     possible

     trends

     in

     the

     AI

     industry

    :
    


    1

    .

     AI

     autonomy

    :

     One

     of

     the

     biggest

     trends

     in

     AI

     is

     toward

     more

     autonomous

     machines

    .

     This

     means

     that

     machines

     will

     be

     able

     to

     make

     decisions

     and

     take

     actions

     without

     human

     intervention

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     systems

    ,

     but

     also

     raises

     questions

     about

     accountability

     and

     control

    .
    


    2

    .

     AI

     ethics

    :

     As

     AI

     becomes

     more

     advanced

    ,

     there

     will

     be

     more

     ethical

     considerations

     to

     take

     into

     account

    .

     We

     will

     need

     to

     ensure

     that

     AI

     systems

     are

     developed

     and

     used

     in

     ways

     that

     respect

     human

     values

     and

     ensure

     that

     they

     are

    



```python
llm.shutdown()
```
