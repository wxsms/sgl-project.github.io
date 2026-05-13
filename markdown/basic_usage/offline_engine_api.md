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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.43it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.42it/s]


    2026-05-13 12:57:15,834 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 12:57:15] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:27,  4.69s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:12,  3.75it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:05<00:04,  8.42it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:05<00:02, 14.26it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=128):  66%|██████▌   | 38/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=112):  66%|██████▌   | 38/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=96):  66%|██████▌   | 38/58 [00:05<00:00, 21.26it/s] 

    Compiling num tokens (num_tokens=80):  66%|██████▌   | 38/58 [00:05<00:00, 21.26it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:05<00:00, 30.29it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:05<00:00, 30.29it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:05<00:00, 30.29it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:05<00:00, 30.29it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:05<00:00, 30.29it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:05<00:00, 30.29it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:05<00:00, 30.29it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:05<00:00, 30.29it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:05<00:00, 30.29it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:05<00:00, 30.29it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:05<00:00, 30.29it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.06 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.03 GB):   3%|▎         | 2/58 [00:00<00:03, 18.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:03, 18.53it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   3%|▎         | 2/58 [00:00<00:03, 18.53it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.02 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.01 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.00 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.73 GB):   9%|▊         | 5/58 [00:00<00:02, 21.77it/s]Capturing num tokens (num_tokens=4096 avail_mem=73.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=3840 avail_mem=73.73 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=3584 avail_mem=73.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=3328 avail_mem=73.02 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.26it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=73.01 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.26it/s]Capturing num tokens (num_tokens=3072 avail_mem=73.01 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=73.01 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=73.01 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=73.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=73.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.00 GB):  22%|██▏       | 13/58 [00:00<00:01, 30.39it/s]Capturing num tokens (num_tokens=1792 avail_mem=73.00 GB):  31%|███       | 18/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.99 GB):  31%|███       | 18/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.99 GB):  31%|███       | 18/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.97 GB):  31%|███       | 18/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=960 avail_mem=72.99 GB):  31%|███       | 18/58 [00:00<00:01, 36.16it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=72.98 GB):  31%|███       | 18/58 [00:00<00:01, 36.16it/s]Capturing num tokens (num_tokens=896 avail_mem=72.98 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=832 avail_mem=72.98 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=768 avail_mem=72.98 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=704 avail_mem=72.98 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=640 avail_mem=72.97 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=576 avail_mem=72.97 GB):  40%|███▉      | 23/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=576 avail_mem=72.97 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.80it/s]Capturing num tokens (num_tokens=512 avail_mem=72.96 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.80it/s]Capturing num tokens (num_tokens=480 avail_mem=72.97 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.80it/s]Capturing num tokens (num_tokens=448 avail_mem=72.97 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.80it/s]Capturing num tokens (num_tokens=416 avail_mem=72.97 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.80it/s]

    Capturing num tokens (num_tokens=384 avail_mem=72.97 GB):  48%|████▊     | 28/58 [00:00<00:00, 41.80it/s]Capturing num tokens (num_tokens=384 avail_mem=72.97 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.72it/s]Capturing num tokens (num_tokens=352 avail_mem=72.96 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.72it/s]Capturing num tokens (num_tokens=320 avail_mem=72.95 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.72it/s]Capturing num tokens (num_tokens=288 avail_mem=72.95 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.72it/s]Capturing num tokens (num_tokens=256 avail_mem=72.95 GB):  57%|█████▋    | 33/58 [00:00<00:00, 43.72it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  57%|█████▋    | 33/58 [00:01<00:00, 43.72it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=224 avail_mem=72.52 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=208 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=192 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.52it/s]

    Capturing num tokens (num_tokens=176 avail_mem=72.51 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=160 avail_mem=72.46 GB):  66%|██████▌   | 38/58 [00:01<00:00, 41.52it/s]Capturing num tokens (num_tokens=160 avail_mem=72.46 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=112 avail_mem=72.22 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.47it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  74%|███████▍  | 43/58 [00:01<00:00, 40.47it/s]Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.12it/s]

    Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  83%|████████▎ | 48/58 [00:01<00:00, 42.12it/s]Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=20 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=16 avail_mem=72.19 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.91it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  91%|█████████▏| 53/58 [00:01<00:00, 42.91it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 43.90it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 39.10it/s]


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
    Generated text:  Yash. I have been a student for over 2 years, and I am a software engineer at Qwen.
    
    I am mainly working on the code for Qwen, which is a GPT model. I am trying to improve its natural language processing abilities.
    
    Do you like to play video games? If so, can you recommend me some video games? If not, please recommend me other types of video games.
    
    I am currently 17 years old and have been studying computer science for the last two years. I have done some programming and have some knowledge of machine learning.
    
    Thank you for your time! Let me know if you
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person, because he is the leader of the country and the head of the government. The president has the power to make all the laws of the country. The president is responsible for making all the important decisions. The president also has the power to appoint the top government officials, such as the Secretary of State, the Secretary of Defense, the Secretary of Energy, the Secretary of Health and Human Services, the Attorney General, and the Chief Justice of the Supreme Court.
    To be elected president, a person must be at least 35 years old and must have been a resident of the United States for at least 14
    ===============================
    Prompt: The capital of France is
    Generated text:  (A) Paris (B) London.
    A. Paris
    
    Paris is the capital of France, and it is located on the Mediterranean Sea in southern France. The French government appoints the president and the prime minister as the leaders of the country, and the president and prime minister are appointed by the European Parliament. The French government is divided into three departments: government departments, lower departments, and executive departments. The government departments manage the government's functions, and the lower departments are responsible for executing the decisions of the government. The executive departments manage day-to-day operations. The president of the French Republic is the head of the government. The
    ===============================
    Prompt: The future of AI is
    Generated text:  very bright. With a huge number of small startups and the massive amounts of data it can process, AI is able to make breakthroughs that were previously impossible. So what are the big ways that AI is going to change the world?
    
    Looking back at how AI has changed the world since 1980, the biggest breakthroughs have been in healthcare, in agriculture, and in education.
    
    Healthcare: More health insights through AI
    
    The healthcare industry has been one of the hardest hit by the development of AI. AI has allowed healthcare providers to be more accurate at diagnosing medical conditions and to predict the likelihood of disease, but it


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, [gender] and [nationality]. I have a [job title] at [company name], and I'm always looking for ways to [describe your job or passion]. I enjoy [mention a hobby or activity you enjoy]. I'm always looking for ways to [describe your goals or aspirations]. What's your favorite hobby or activity? I'm always looking for ways to [describe your goals or aspirations]. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, which is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many world-renowned museums, including the Musée d'Orsay and the Musée Rodin. Paris is also known for its diverse cuisine, including French cuisine, as well as its wine and wine-making industry. The city is also home to many international organizations and institutions, including the European Parliament and the European Central Bank. Paris is a bustling and vibrant city with a rich history and culture, making it
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be increased pressure to consider the ethical implications of its use. This could lead to more stringent regulations and standards for AI development and deployment
    


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
    Generated text:  [insert name] and I am a [insert character's profession or role] who has been around for [insert number] years. I was born in [insert birthplace], where I was brought up by [insert family members or caregivers]. I am a [insert occupation] who has dedicated my life to [insert occupation or cause]. I am always motivated to make a positive impact in the world and I am dedicated to using my skills and experience to do so. I am always looking for new challenges and opportunities to learn and grow. I am a [insert personality trait or quality] and I am always eager to share my experiences and
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    This statement is true and correct. It accurately describes the geographical and historical significance of Paris, which is the capital city of France. Paris is the largest city in Europe and one of the most visited cities in the world, known for its artistic, cultural, and historical attractions. Paris is also the seat of the French government and a major international metropolis with a diverse population, rich history, and a rich cultural heritage. Paris' status as the world's most visited city is attributed to its unique blend of classical and modern architectural styles, vibrant street life, and flourishing arts and culture, which make it a popular destination for tourists
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly complex and uncertain, with potential applications in many areas of human society. Some of the possible future trends in AI include:
    
    1. Advancements in deep learning: As the neural networks become increasingly complex, we may see improvements in the accuracy of AI systems, making them more effective at solving complex problems.
    
    2. Increased focus on ethics and privacy: As more AI systems are used in decision-making, there may be an increased focus on ethical considerations and privacy concerns. This may lead to more stringent regulations and ethical guidelines for AI systems.
    
    3. Globalization of AI: As AI becomes more integrated into our daily lives, it may become even


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

     writer

    .

     I

     started

     writing

     stories

     when

     I

     was

     a

     teenager

     and

     have

     been

     trying

     to

     write

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

     exploring

     different

     genres

     and

     styles

    ,

     from

     science

     fiction

     and

     romance

     to

     fantasy

     and

     historical

     fiction

    .

     I

     like

     to

     create

     characters

     that

     are

     rich

     and

     well

    -develop

    ed

    ,

     and

     I

    'm

     always

     looking

     for

     new

     and

     exciting

     ideas

     to

     write

     about

    .

     I

    'm

     looking

     forward

     to

     meeting

     you

    ,

     and

     I

     hope

     we

     can

     have

     a

     conversation

     about

     your

     own

     writing

     adventures

    !

     #

    Welcome

    To

    The

    Writing

    World

     #

    Author

     #

    Self

    Introduction

     #

    Writing

    Life

     #

    Creative

    M

    inds

     #

    Char

    l

    ies

    Writing

    Community

    .

     Hey

    !

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    The

     French

     capital

     city

     is

     Paris

    .

     It

     is

     the

     largest

     city

     and

     the

     seat

     of

     the

     Government

     of

     France

    ,

     and

     is

     situated

     on

     the

     Se

    ine

     river

     in

     the

     south

     of

     the

     country

    .

     It

     is

     also

     known

     as

     "

    la

     Hay

    e

    "

     (

    the

     land

     of

     light

    ).

     The

     city

     has

     a

     population

     of

     around

     

    2

    .

    3

     million

     people

    .

     It

     is

     home

     to

     several

     iconic

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

     and

     the

     Lou

    vre

     Museum

    .

     Paris

     is

     also

     known

     for

     its

     diverse

     cultural

     and

     artistic

     scene

    ,

     including

     the

     many

     museums

    ,

     theaters

    ,

     and

     restaurants

    .

     The

     city

     is

     a

     major

     hub

     for

     business

    ,

     trade

    ,

     and

     international

     affairs

    .

     Paris

    's

     vibrant

     lifestyle

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     a

     rapidly

     evolving

     field

    ,

     and

     there

     are

     many

     potential

     trends

     that

     are

     shaping

     the

     direction

     of

     AI

     development

    .

     Some

     of

     the

     possible

     future

     trends

     in

     AI

     include

    :
    


    1

    .

     Increasing

     reliance

     on

     AI

     for

     decision

    -making

    :

     As

     AI

     technology

     becomes

     more

     advanced

     and

     accurate

    ,

     it

     is

     likely

     that

     decision

    -making

     will

     increasingly

     rely

     on

     AI

     to

     make

     choices

     and

     decisions

    .
    


    2

    .

     Enhanced

     integration

     of

     AI

     with

     other

     technologies

    :

     AI

     is

     currently

     limited

     to

     a

     few

     specific

     applications

    ,

     but

     as

     more

     applications

     of

     AI

     become

     possible

    ,

     we

     can

     expect

     that

     they

     will

     become

     increasingly

     integrated

     with

     other

     technologies

    .
    


    3

    .

     Greater

     focus

     on

     ethical

     considerations

    :

     As

     more

     AI

     systems

     become

     more

     advanced

    ,

     there

     will

     be

     a

    



```python
llm.shutdown()
```
