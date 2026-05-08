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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.56it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.56it/s]


    2026-05-08 05:45:33,220 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 05:45:33] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:47,  3.99s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:47,  3.99s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:47,  3.99s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:47,  3.99s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:42,  1.27it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:42,  1.27it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:42,  1.27it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:42,  1.27it/s]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:42,  1.27it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:04<00:16,  3.05it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:04<00:16,  3.05it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:04<00:16,  3.05it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:04<00:16,  3.05it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:04<00:16,  3.05it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:04<00:16,  3.05it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:04<00:16,  3.05it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:04<00:06,  6.53it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:04<00:06,  6.53it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:04<00:06,  6.53it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:04<00:06,  6.53it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:04<00:06,  6.53it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:04<00:06,  6.53it/s]

    Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:04<00:06,  6.53it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03, 10.76it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03, 10.76it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03, 10.76it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03, 10.76it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03, 10.76it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03, 10.76it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:03, 10.76it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:03, 10.76it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:01, 16.46it/s]

    Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:04<00:01, 16.46it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:04<00:00, 24.01it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:04<00:00, 24.01it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:04<00:00, 24.01it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:04<00:00, 24.01it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:04<00:00, 24.01it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:04<00:00, 24.01it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:04<00:00, 24.01it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:04<00:00, 24.01it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 30.50it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 30.50it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 30.50it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 30.50it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 30.50it/s]

    Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 30.50it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 30.50it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 30.50it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 30.50it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 38.60it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 38.60it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 38.60it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 38.60it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 38.60it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 38.60it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 38.60it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 38.60it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 38.60it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=59.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.10 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=59.10 GB):   3%|▎         | 2/58 [00:00<00:03, 14.29it/s]Capturing num tokens (num_tokens=7168 avail_mem=59.09 GB):   3%|▎         | 2/58 [00:00<00:03, 14.29it/s]Capturing num tokens (num_tokens=6656 avail_mem=59.08 GB):   3%|▎         | 2/58 [00:00<00:03, 14.29it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=59.08 GB):   7%|▋         | 4/58 [00:00<00:03, 16.81it/s]Capturing num tokens (num_tokens=6144 avail_mem=59.08 GB):   7%|▋         | 4/58 [00:00<00:03, 16.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=59.08 GB):   7%|▋         | 4/58 [00:00<00:03, 16.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.07 GB):   7%|▋         | 4/58 [00:00<00:03, 16.81it/s]Capturing num tokens (num_tokens=5120 avail_mem=59.07 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=4608 avail_mem=59.06 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=59.06 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=59.06 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.06it/s]Capturing num tokens (num_tokens=3584 avail_mem=59.05 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.06it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=59.05 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.12it/s]Capturing num tokens (num_tokens=3328 avail_mem=59.05 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.12it/s]Capturing num tokens (num_tokens=3072 avail_mem=59.05 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.12it/s]Capturing num tokens (num_tokens=2816 avail_mem=59.05 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=59.04 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.04 GB):  19%|█▉        | 11/58 [00:00<00:01, 27.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=59.04 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.70it/s]Capturing num tokens (num_tokens=2048 avail_mem=59.03 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.70it/s]Capturing num tokens (num_tokens=1792 avail_mem=59.03 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.70it/s]Capturing num tokens (num_tokens=1536 avail_mem=59.03 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.70it/s]Capturing num tokens (num_tokens=1280 avail_mem=59.03 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.70it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=59.01 GB):  28%|██▊       | 16/58 [00:00<00:01, 33.70it/s]Capturing num tokens (num_tokens=1024 avail_mem=59.01 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=960 avail_mem=59.02 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.99it/s] Capturing num tokens (num_tokens=896 avail_mem=59.02 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=832 avail_mem=59.01 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=768 avail_mem=59.01 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=704 avail_mem=59.01 GB):  36%|███▌      | 21/58 [00:00<00:01, 32.99it/s]Capturing num tokens (num_tokens=704 avail_mem=59.01 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.86it/s]Capturing num tokens (num_tokens=640 avail_mem=59.00 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.86it/s]Capturing num tokens (num_tokens=576 avail_mem=59.00 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.86it/s]Capturing num tokens (num_tokens=512 avail_mem=58.99 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.86it/s]

    Capturing num tokens (num_tokens=480 avail_mem=59.00 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.86it/s]Capturing num tokens (num_tokens=448 avail_mem=59.00 GB):  45%|████▍     | 26/58 [00:00<00:00, 35.86it/s]Capturing num tokens (num_tokens=448 avail_mem=59.00 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.43it/s]Capturing num tokens (num_tokens=416 avail_mem=59.00 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.43it/s]Capturing num tokens (num_tokens=384 avail_mem=59.00 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.43it/s]Capturing num tokens (num_tokens=352 avail_mem=58.99 GB):  53%|█████▎    | 31/58 [00:00<00:00, 39.43it/s]Capturing num tokens (num_tokens=320 avail_mem=58.99 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.43it/s]Capturing num tokens (num_tokens=288 avail_mem=58.99 GB):  53%|█████▎    | 31/58 [00:01<00:00, 39.43it/s]Capturing num tokens (num_tokens=288 avail_mem=58.99 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=256 avail_mem=58.98 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=240 avail_mem=58.98 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=224 avail_mem=58.98 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]

    Capturing num tokens (num_tokens=208 avail_mem=58.97 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=192 avail_mem=58.97 GB):  62%|██████▏   | 36/58 [00:01<00:00, 42.03it/s]Capturing num tokens (num_tokens=192 avail_mem=58.97 GB):  71%|███████   | 41/58 [00:01<00:00, 44.04it/s]Capturing num tokens (num_tokens=176 avail_mem=58.97 GB):  71%|███████   | 41/58 [00:01<00:00, 44.04it/s]Capturing num tokens (num_tokens=160 avail_mem=58.97 GB):  71%|███████   | 41/58 [00:01<00:00, 44.04it/s]Capturing num tokens (num_tokens=144 avail_mem=58.96 GB):  71%|███████   | 41/58 [00:01<00:00, 44.04it/s]Capturing num tokens (num_tokens=128 avail_mem=58.96 GB):  71%|███████   | 41/58 [00:01<00:00, 44.04it/s]Capturing num tokens (num_tokens=112 avail_mem=58.96 GB):  71%|███████   | 41/58 [00:01<00:00, 44.04it/s]Capturing num tokens (num_tokens=112 avail_mem=58.96 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=96 avail_mem=58.95 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.15it/s] Capturing num tokens (num_tokens=80 avail_mem=58.95 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=64 avail_mem=58.95 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.15it/s]

    Capturing num tokens (num_tokens=48 avail_mem=58.94 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=32 avail_mem=58.94 GB):  79%|███████▉  | 46/58 [00:01<00:00, 45.15it/s]Capturing num tokens (num_tokens=32 avail_mem=58.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.70it/s]Capturing num tokens (num_tokens=28 avail_mem=58.93 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.70it/s]Capturing num tokens (num_tokens=24 avail_mem=58.93 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.70it/s]Capturing num tokens (num_tokens=20 avail_mem=58.93 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.70it/s]Capturing num tokens (num_tokens=16 avail_mem=58.93 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.70it/s]Capturing num tokens (num_tokens=12 avail_mem=58.92 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.70it/s]Capturing num tokens (num_tokens=12 avail_mem=58.92 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.45it/s]Capturing num tokens (num_tokens=8 avail_mem=58.92 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.45it/s] Capturing num tokens (num_tokens=4 avail_mem=58.92 GB):  97%|█████████▋| 56/58 [00:01<00:00, 44.45it/s]Capturing num tokens (num_tokens=4 avail_mem=58.92 GB): 100%|██████████| 58/58 [00:01<00:00, 37.83it/s]


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
    Generated text:  Li Ling, and I am a high school student. I love reading, watching TV, and playing sports. My favorite subject is history because it gives me a lot of knowledge about China's history. I am always eager to learn more and share my knowledge with others.
    What would you say to Li Ling?
    
    Li Ling, thank you for your kind words. I'm glad you like reading and watching TV. In my opinion, history is a subject that is worth learning because it provides us with knowledge about China's history. I am always eager to learn more and share my knowledge with others. I hope you can learn more about history too
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to have in the country. The number of bases can range from 1 to 100. The president has a unique strategy for selecting the number of bases.
    
    The strategy is as follows:
    
    1. He considers a set of numbers from 1 to 100, and for each number, he checks if it is a prime number.
    2. For each prime number, he adds 1 to it and checks if the resulting number is a prime number.
    3. If all numbers from 1 to 100 pass this process, he chooses the number of bases that results in
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the largest and most populous city of France, and is the capital of the French departments of Paris, Île-de-France, and Île-Îles, and of the prefecture of the Île-de-France. It is also one of the four official languages of the European Union, and a member of the United Nations, the Council of Europe, and the Organisation of American States. The population of the city proper is about 2.1 million, while the population of the Greater Parisian area (including its environs) is about 4.5 million. Paris is considered the cultural,
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the people
    The world has a profound impact on the history of the world, for example, the changes that the Industrial Revolution brought about on the world, or the impact that the Industrial Revolution brought about on the world. The changes brought about by the industrial revolution have left a mark on the history of the world. The future of AI will be influenced by the changes brought about by the future of the world.
    AI has become an integral part of daily life in the world. Artificial intelligence (AI) has the capability to analyze data and generate results in a human-like manner, which makes it highly capable. AI has been


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a short, positive description of your personality or skills]. I enjoy [insert a short, positive description of your hobbies or interests]. What do you like to do in your free time? I love [insert a short, positive description of your hobbies or interests]. I'm always looking for new experiences and challenges to try. What's your favorite hobby or activity? I love [insert a short, positive description of your favorite hobby or
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history and a vibrant culture, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also home to many famous museums, including the Louvre, the Musée d'Orsay, and the Musée Rodin. The city is also known for its cuisine, including its famous French fries and its traditional French wine. Paris is a popular tourist destination and a cultural hub for France and the world. It is a city that is steeped in history and culture, and is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way that AI is used and developed. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI, there is a growing emphasis on ethical AI. This includes developing AI that is designed to be transparent, accountable, and fair, and that is used to address social and environmental issues.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced
    


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
    Generated text:  [Name] and I am [Age] years old. I enjoy [something I like doing], [something I like learning about], or [something I like doing outdoors. ]. I am currently [where you are] and [what you do for a living]. If you have any questions about my background, education, or experiences, please feel free to ask. Let me know if you would like me to elaborate on anything. Hello, my name is [Name] and I am [Age] years old. I enjoy [something I like doing], [something I like learning about], or [something I like doing outdoors. ]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located on the Seine river in the North of the country. It was founded in the 8th century by Charlemagne and became the seat of the French monarchy in the 13th century, and is known as the "City of Love" for its many museums and historical landmarks. The city is home to the Eiffel Tower, the Louvre Museum, and the Arc de Triomphe. Paris is the third most populous city in Europe, with around 2. 2 million residents. In addition to the historical center, the city is also home to many trendy neighborhoods and vibrant art scenes. 
    
    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by rapid technological advancements, shifting industries, and an increasing reliance on data and algorithms. Some of the possible future trends in AI include:
    
    1. Increased automation: AI will continue to be a primary tool for automating routine tasks, such as data analysis, customer service, and manufacturing. As more industries become automated, there will be a need for AI systems to handle higher levels of automation.
    
    2. Integration of AI and human decision-making: AI systems will continue to become more integrated with human decision-making, allowing humans to rely on AI to make decisions when necessary.
    
    3. Increased focus on privacy and ethics: There will


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

     occupation

    ]

     with

     [

    Your

     experience

    ,

     skills

    ,

     and

     background

    ].

     I

     love

     [

    Your

     interest

     or

     passion

    ],

     and

     I

    'm

     passionate

     about

     [

    Your

     career

     or

     personal

     goal

    ].

     I

     strive

     to

     make

     a

     positive

     impact

     on

     the

     world

    ,

     and

     I

     believe

     that

     my

     skills

     and

     experience

     can

     help

     make

     a

     difference

     in

     the

     lives

     of

     others

    .

     
    


    What

     is

     one

     thing

     you

     can

    't

     stand

     about

     yourself

    ?

     How

     would

     you

     change

     it

    ?

     As

     an

     AI

     language

     model

    ,

     I

     don

    't

     have

     personal

     experiences

     or

     emotions

    ,

     but

     I

     can

     tell

     you

     that

     I

     don

    't

     have

     a

     personal

     interest

     or

     passion

    .

     However

    ,

     I

    'm

     designed

     to

     help

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    That

    's

     correct

    !

     Paris

     is

     the

     capital

     of

     France

     and

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

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     famous

     for

     its

     French

     cuisine

    ,

     arts

    ,

     and

     fashion

    .

     Paris

     is

     home

     to

     many

     iconic

     landmarks

     and

     is

     a

     world

    -ren

    owned

     city

     with

     a

     rich

     history

     and

     culture

    .

     The

     city

     is

     also

     known

     for

     its

     annual

     E

    iff

    el

     Tower

     Par

    c

     de

     la

     Col

    ô

    mb

    ienne

     and

     the

     F

    ête

     de

     la

     Saint

    -J

    ean

    -B

    apt

    iste

    ,

     which

     celebrates

     the

     French

     Revolution

    .

     Paris

     is

     a

     popular

     tourist

     destination

     and

     is

     home

     to

     many

     world

    -ren

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     full

     of

     possibilities

    ,

     and

     there

     are

     many

     different

     trends

     that

     could

     shape

     the

     way

     the

     technology

     evolves

    .

     Here

     are

     some

     potential

     future

     trends

     that

     could

     be

     explored

    :
    


    1

    .

     Increased

     emphasis

     on

     ethical

     considerations

    :

     As

     more

     and

     more

     AI

     systems

     are

     developed

    ,

     it

    's

     likely

     that

     we

    'll

     see

     an

     increase

     in

     discussions

     about

     the

     ethics

     of

     AI

    .

     This

     could

     lead

     to

     new

     regulations

     and

     guidelines

     that

     govern

     the

     use

     of

     AI

     in

     various

     industries

    .
    


    2

    .

     Integration

     with

     human

     creativity

    :

     As

     AI

     continues

     to

     advance

    ,

     it

    's

     possible

     that

     we

    'll

     see

     even

     more

     integration

     with

     human

     creativity

    .

     AI

    -powered

     tools

     could

     be

     used

     to

     assist

     with

     creative

     writing

    ,

     visual

     art

    ,

     and

     other

     forms

     of

    



```python
llm.shutdown()
```
