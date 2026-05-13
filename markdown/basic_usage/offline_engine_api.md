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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.38it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.37it/s]


    2026-05-13 06:38:16,633 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-13 06:38:16] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:27,  4.70s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:38,  1.38it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:11,  4.14it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  4.14it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:03,  9.32it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:03,  9.32it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:03,  9.32it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:03,  9.32it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:03,  9.32it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:03,  9.32it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.32it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.32it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.32it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.32it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.05it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.05it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.05it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.05it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.05it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.05it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.05it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.05it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.05it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.05it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.05it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.81it/s] 

    Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.81it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 29.29it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 29.29it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 35.05it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 35.05it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.27 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.27 GB):   3%|▎         | 2/58 [00:00<00:03, 16.20it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 16.20it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 16.20it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   7%|▋         | 4/58 [00:00<00:03, 17.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   7%|▋         | 4/58 [00:00<00:03, 17.60it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   7%|▋         | 4/58 [00:00<00:03, 17.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   7%|▋         | 4/58 [00:00<00:03, 17.60it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.42it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  12%|█▏        | 7/58 [00:00<00:02, 21.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  19%|█▉        | 11/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.14it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.14it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.14it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.14it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.02it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.02it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.02it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.02it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.02it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  36%|███▌      | 21/58 [00:00<00:00, 38.02it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.05it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.05it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.05it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.05it/s]

    Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.05it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  45%|████▍     | 26/58 [00:00<00:00, 41.05it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=320 avail_mem=75.99 GB):  53%|█████▎    | 31/58 [00:00<00:00, 43.49it/s]Capturing num tokens (num_tokens=288 avail_mem=75.99 GB):  53%|█████▎    | 31/58 [00:01<00:00, 43.49it/s]Capturing num tokens (num_tokens=288 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.77it/s]Capturing num tokens (num_tokens=256 avail_mem=75.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.77it/s]

    Capturing num tokens (num_tokens=240 avail_mem=75.98 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.77it/s]Capturing num tokens (num_tokens=224 avail_mem=75.98 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.77it/s]Capturing num tokens (num_tokens=208 avail_mem=75.97 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.77it/s]Capturing num tokens (num_tokens=192 avail_mem=75.48 GB):  62%|██████▏   | 36/58 [00:01<00:00, 38.77it/s]Capturing num tokens (num_tokens=192 avail_mem=75.48 GB):  71%|███████   | 41/58 [00:01<00:00, 36.98it/s]Capturing num tokens (num_tokens=176 avail_mem=75.48 GB):  71%|███████   | 41/58 [00:01<00:00, 36.98it/s]Capturing num tokens (num_tokens=160 avail_mem=75.38 GB):  71%|███████   | 41/58 [00:01<00:00, 36.98it/s]Capturing num tokens (num_tokens=144 avail_mem=75.31 GB):  71%|███████   | 41/58 [00:01<00:00, 36.98it/s]Capturing num tokens (num_tokens=128 avail_mem=75.31 GB):  71%|███████   | 41/58 [00:01<00:00, 36.98it/s]Capturing num tokens (num_tokens=112 avail_mem=75.31 GB):  71%|███████   | 41/58 [00:01<00:00, 36.98it/s]

    Capturing num tokens (num_tokens=112 avail_mem=75.31 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=96 avail_mem=75.30 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.09it/s] Capturing num tokens (num_tokens=80 avail_mem=75.30 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=64 avail_mem=75.30 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=48 avail_mem=75.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=32 avail_mem=75.29 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.09it/s]Capturing num tokens (num_tokens=32 avail_mem=75.29 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=28 avail_mem=75.28 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=24 avail_mem=75.28 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=20 avail_mem=75.28 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=16 avail_mem=75.28 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.09it/s]Capturing num tokens (num_tokens=12 avail_mem=75.27 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.09it/s]

    Capturing num tokens (num_tokens=12 avail_mem=75.27 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=8 avail_mem=75.27 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.02it/s] Capturing num tokens (num_tokens=4 avail_mem=75.27 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.02it/s]Capturing num tokens (num_tokens=4 avail_mem=75.27 GB): 100%|██████████| 58/58 [00:01<00:00, 37.70it/s]


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
    Generated text:  Alicia and I'm writing a short story. I'm trying to come up with a title and I'm a little unsure. I've been working on it for a while now and I feel it should be something simple and approachable for a young adult audience. 
    
    What title would you give for this story? "The Talented Girl," "The Thrifty Girl," "The Well-Entrenched Girl," or "The Charming Girl"? I'm not sure which one would be most appropriate for a young adult audience. Can you help me choose a title that aligns with the mood and tone of the story?
    
    Sure, I can
    ===============================
    Prompt: The president of the United States is
    Generated text:  the leader of which of the following? A. The American people B. The American government C. The American military D. The American people and the American government
    Answer: B
    
    The more commonly used type of bridge that spans the river in Shanghai is ____.
    A. Suspension bridge
    B. Steel girder bridge
    C. Arch bridge
    D. Floating bridge
    Answer: B
    
    On a certain street in Beijing, there is a park called South Lake. Among the following options, which park is located in the same administrative area as South Lake?
    A. Xiaoshan
    B. Hengqin
    C. Be
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. For the first time, Paris will be an international city again, starting next year. What did the person do for the first time? - read newspapers - learned a new language - played sports - go hiking - go to school
    The answer to the riddle is read newspapers. The person read newspapers for the first time. The answer can be found by analyzing each option:
    
    - Read newspapers: The given information states that Paris will be an international city again starting next year, and the person is already aware of this change. This directly aligns with the person reading newspapers.
    
    - Learned a new language: This is a related task
    ===============================
    Prompt: The future of AI is
    Generated text:  here. Whether you are an AI enthusiast or a beginner in the field, you are going to need to learn about the fundamental concepts and theories of AI. In this blog post, we will be discussing the difference between AI and machine learning. We will also explore the different types of AI and what each one is for. Additionally, we will delve into the history of AI and the development of the field. In this blog post, we will also discuss the future of AI and how it is likely to evolve in the coming years. Let’s get started!\nIn a world where artificial intelligence has become an integral part of our daily lives,


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


    Generated text:  [Name] and I am a [Age] year old [Occupation]. I am a [Type of Vehicle] [Vehicle Name]. I have been driving for [Number of Years] years and I have [Number of Miles] miles on my [Vehicle Name]. I am [Gender] and I have [Number of Children] children. I am [Occupation] and I enjoy [Favorite Activity/Interest]. I am [Age] years old and I am [Occupation]. I am [Name] and I am a [Age] year old [Occupation]. I am a [Type of Vehicle] [Vehicle Name].
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French National Library, and the French National Opera. Paris is a cultural and economic hub, known for its rich history, art, and cuisine. It is also home to many famous landmarks and attractions, including the Eiffel Tower, the Louvre, and the Champs-Élysées. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. It is a city that has played a significant role in French history and continues
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to help diagnose and treat diseases, and it has the potential to revolutionize the field. AI-powered diagnostic tools, such as machine learning algorithms, are being developed to help doctors make more accurate diagnoses and to predict patient outcomes.
    
    2. AI in finance: AI is already being used in finance to help with fraud detection, risk management, and portfolio optimization. As AI technology continues to improve, we can expect to see even
    


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
    Generated text:  [Name], and I am a [job title] who has been [number of years] in the industry. I have a keen interest in [mention specific area of interest, such as [industry, customer service, etc.]]. I love [specific hobby or activity you enjoy doing, like [reading, playing sports, etc.]]. What do you think about [something in particular about your job or interests]?
    [Name]: I'm glad to meet you! As a [job title] with 5 years of experience, I have a keen interest in [mention specific area of interest, such as [industry, customer service,
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as the City of Light, and is the seat of government, administration, culture, commerce, and society in the country.
    
    That statement is factually correct. However, if you're interested in lesser-known facts about Paris, such as its cuisine, art scene, or history, you might find some interesting tidbits. For example:
    
    * Paris is home to the Louvre Museum, one of the world's largest and most famous art galleries.
    * The Eiffel Tower is considered one of the seven wonders of the world.
    * The Paris Basin, a vast area of the city bounded by the Seine River,
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid development and innovation, with new technologies and applications emerging at a rapid pace. Here are some possible future trends in AI:
    
    1. Increased integration with other fields: As AI becomes more integrated with other fields such as healthcare, energy, and transportation, it is expected to have a significant impact on these sectors.
    
    2. Greater emphasis on ethical considerations: As AI becomes more sophisticated, there will be increasing scrutiny around its impact on human lives and society as a whole. There will be a greater emphasis on ethical considerations and best practices for developing and deploying AI systems.
    
    3. Greater automation of certain tasks: As AI becomes


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

     am

     a

     computer

     scientist

    .

     I

     recently

     graduated

     from

     [

    University

    ]

     and

     have

     been

     working

     in

     [

    field

     of

     interest

    ].

     I

     am

     passionate

     about

     [

    disc

    ipline

     or

     area

     of

     interest

    ]

     and

     I

     am

     always

     looking

     for

     new

     challenges

    .

     I

     enjoy

     [

    personal

     interest

     or

     hobby

    ],

     and

     I

    'm

     always

     exploring

     new

     ideas

     and

     technologies

    .

     What

     exc

    ites

     me

     the

     most

     is

     [

    exc

    it

    ement

     or

     interest

    ].

     I

     am

     excited

     to

     join

     your

     team

     and

     work

     with

     you

     to

     achieve

     your

     goals

    !

     [

    Name

    ],

     this

     is

     [

    Name

    ]

     and

     I

     am

     excited

     to

     meet

     you

    !

     
    


    ---
    


    What

     other

     information

     can

     you

     provide

     about

     [

    Name

    ]

     and

     why

     they

     are

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     left

     bank

     of

     the

     Se

    ine

     River

     in

     the

     north

    western

     part

     of

     the

     country

    ,

     and

     is

     the

     largest

     city

     in

     both

     France

     and

     Europe

    .

     It

     is

     known

     for

     its

     historical

     landmarks

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

     Notre

     Dame

     Cathedral

    ,

     and

     Ch

    amps

    -

    É

    lys

    ées

    .

     The

     city

     is

     also

     renowned

     for

     its

     fashion

     and

     gastr

    onomy

    ,

     which

     has

     been

     a

     significant

     part

     of

     its

     cultural

     identity

     since

     the

     

    1

    9

    th

     century

    .

     Paris

     is

     a

     popular

     tourist

     destination

     with

     millions

     of

     visitors

     annually

     and

     is

     considered

     a

     cultural

     and

     economic

     powerhouse

     in

     France

    .

     Its

     skyline

     is

     one

     of

     the

     most

     impressive

     in

     Europe

    ,

     and

     its

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

    ,

     and

     there

     are

     many

     possibilities

     ahead

     for

     how

     AI

     will

     evolve

    .

     Here

     are

     some

     possible

     trends

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     will

     increasingly

     be

     integrated

     with

     other

     technologies

    ,

     such

     as

     quantum

     computing

    ,

     neural

     networks

    ,

     and

     blockchain

    ,

     to

     create

     new

    ,

     powerful

     systems

     that

     can

     solve

     complex

     problems

     that

     are

     beyond

     the

     scope

     of

     current

     AI

    .
    


    2

    .

     More

     diverse

    ,

     ethical

    ,

     and

     responsible

     AI

    :

     As

     AI

     becomes

     more

     prevalent

    ,

     it

     will

     become

     more

     important

     to

     consider

     the

     ethical

     implications

     and

     make

     sure

     that

     AI

     systems

     are

     designed

     to

     be

     transparent

    ,

     accountable

    ,

     and

     responsible

    .

     This

     includes

     ensuring

     that

     AI

     systems

     are

     transparent

     about

     how

     they

     make

     decisions

    ,

     that

     they

    



```python
llm.shutdown()
```
