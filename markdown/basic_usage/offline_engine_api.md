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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.81it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.81it/s]


    2026-05-12 06:54:49,616 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-12 06:54:49] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:01,  4.24s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:04<00:11,  4.12it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:04,  9.21it/s]

    Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:02, 14.40it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 20.85it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 20.85it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 20.85it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 20.85it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 20.85it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:01, 20.85it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:01, 20.85it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:01, 20.85it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:01, 20.85it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:04<00:00, 28.09it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:04<00:00, 28.09it/s]

    Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:04<00:00, 28.09it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:04<00:00, 28.09it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:04<00:00, 28.09it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:04<00:00, 28.09it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:04<00:00, 28.09it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:04<00:00, 28.09it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:04<00:00, 28.09it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 35.81it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 35.81it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 35.81it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 35.81it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 35.81it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 35.81it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 35.81it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.56 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.53 GB):   3%|▎         | 2/58 [00:00<00:03, 18.15it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.10 GB):   3%|▎         | 2/58 [00:00<00:03, 18.15it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.09 GB):   3%|▎         | 2/58 [00:00<00:03, 18.15it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.09 GB):   3%|▎         | 2/58 [00:00<00:03, 18.15it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.09 GB):   9%|▊         | 5/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.09 GB):   9%|▊         | 5/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.08 GB):   9%|▊         | 5/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.07 GB):   9%|▊         | 5/58 [00:00<00:02, 20.90it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.18it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.18it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.07 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.18it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=72.06 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.06 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.18it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.50it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.50it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.06 GB):  21%|██        | 12/58 [00:00<00:01, 28.50it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.05 GB):  21%|██        | 12/58 [00:00<00:01, 28.50it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.05 GB):  21%|██        | 12/58 [00:00<00:01, 28.50it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.05 GB):  21%|██        | 12/58 [00:00<00:01, 28.50it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.66it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.66it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=72.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.66it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.02 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.66it/s]Capturing num tokens (num_tokens=960 avail_mem=72.03 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.66it/s] Capturing num tokens (num_tokens=960 avail_mem=72.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=896 avail_mem=72.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=832 avail_mem=72.03 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=768 avail_mem=72.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=704 avail_mem=72.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=640 avail_mem=72.02 GB):  38%|███▊      | 22/58 [00:00<00:00, 38.39it/s]Capturing num tokens (num_tokens=640 avail_mem=72.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=576 avail_mem=72.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=512 avail_mem=72.00 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.29it/s]

    Capturing num tokens (num_tokens=480 avail_mem=72.02 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=448 avail_mem=72.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=416 avail_mem=72.01 GB):  47%|████▋     | 27/58 [00:00<00:00, 41.29it/s]Capturing num tokens (num_tokens=416 avail_mem=72.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=384 avail_mem=72.01 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=352 avail_mem=72.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=320 avail_mem=72.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=288 avail_mem=72.00 GB):  55%|█████▌    | 32/58 [00:00<00:00, 42.01it/s]Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  55%|█████▌    | 32/58 [00:01<00:00, 42.01it/s]Capturing num tokens (num_tokens=256 avail_mem=71.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=240 avail_mem=71.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.69it/s]

    Capturing num tokens (num_tokens=224 avail_mem=71.99 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=208 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=192 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=176 avail_mem=71.98 GB):  64%|██████▍   | 37/58 [00:01<00:00, 40.69it/s]Capturing num tokens (num_tokens=176 avail_mem=71.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=160 avail_mem=71.98 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=144 avail_mem=71.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.25it/s]

    Capturing num tokens (num_tokens=128 avail_mem=71.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=112 avail_mem=71.97 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.25it/s]Capturing num tokens (num_tokens=96 avail_mem=71.96 GB):  72%|███████▏  | 42/58 [00:01<00:00, 41.25it/s] Capturing num tokens (num_tokens=96 avail_mem=71.96 GB):  81%|████████  | 47/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=80 avail_mem=71.96 GB):  81%|████████  | 47/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=64 avail_mem=71.96 GB):  81%|████████  | 47/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=48 avail_mem=71.95 GB):  81%|████████  | 47/58 [00:01<00:00, 33.42it/s]Capturing num tokens (num_tokens=32 avail_mem=71.95 GB):  81%|████████  | 47/58 [00:01<00:00, 33.42it/s]

    Capturing num tokens (num_tokens=32 avail_mem=71.95 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.84it/s]Capturing num tokens (num_tokens=28 avail_mem=71.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.84it/s]Capturing num tokens (num_tokens=24 avail_mem=71.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.84it/s]Capturing num tokens (num_tokens=20 avail_mem=71.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.84it/s]Capturing num tokens (num_tokens=16 avail_mem=71.94 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.84it/s]Capturing num tokens (num_tokens=12 avail_mem=71.93 GB):  88%|████████▊ | 51/58 [00:01<00:00, 32.84it/s]Capturing num tokens (num_tokens=12 avail_mem=71.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.69it/s]Capturing num tokens (num_tokens=8 avail_mem=71.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.69it/s] Capturing num tokens (num_tokens=4 avail_mem=71.93 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.69it/s]Capturing num tokens (num_tokens=4 avail_mem=71.93 GB): 100%|██████████| 58/58 [00:01<00:00, 34.77it/s]


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
    Generated text:  Yuri Vasilievich. I am a professional industrial designer, and I have a passion for creating modern designs that are suitable for both industrial and residential uses. I specialize in the design of electrical, lighting, automation, and interior design. My designs are not only aesthetically pleasing but also functional and effective in achieving desired outcomes.
    As an industrial designer, I believe that the purpose of a design is to meet the needs of the user. I work closely with architects and engineers to provide custom solutions that meet their specific requirements. I am always up-to-date with the latest design trends and technologies to ensure that my designs are contemporary and relevant
    ===============================
    Prompt: The president of the United States is
    Generated text:  an elected position. Which of the following statements about the president is true?
    
    A: The president of the United States is the highest-ranking government official.
    
    B: The president of the United States is the head of the executive branch of the government.
    
    C: The president of the United States represents the interests of the American people.
    
    D: The president of the United States can veto legislation and sign laws into law. The president of the United States is indeed an elected position. Let's analyze each statement one by one to determine which one is true:
    
    A: The president of the United States is the highest-ranking government official.
    - This statement is
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. If you want to visit the Louvre Museum, you should go there. (判断题)
    A. 错误
    B. 正确
    答案:
    A
    
    急性心肌梗死的手术指征是____
    A. 38岁
    B. 20岁以上
    C. 30岁以下
    D. 30岁以上
    E. 20岁以下
    答案:
    B
    
    6.急性心肌梗死患者在急性期绝对卧床，正确的卧位为____
    A. 患侧卧位
    B. 高半坐卧位
    C
    ===============================
    Prompt: The future of AI is
    Generated text:  on a collision course with the future of human life and it is time to reevaluate the nature of this future. The future of AI and the human future is one that the human race cannot afford to ignore.
    The future of AI is a future of diversity, growth and change. The future of human life is one that is characterized by increased access to information, greater levels of access to technology, and greater empathy towards the less fortunate. The future of AI is one that is characterized by the creation of intelligent machines that can learn, adapt and make decisions based on context and feedback.
    The future of AI is one that is characterized by the development


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Person] who is [Describe your personality traits here]. I enjoy [List three hobbies or interests]. I'm [Describe your physical appearance here]. I'm [Describe your physical appearance here]. I'm [Describe your physical appearance here]. I'm [Describe your physical appearance here]. I'm [Describe your physical appearance here]. I'm [Describe your physical appearance here]. I'm [Describe your physical appearance here]. I'm [Describe your physical appearance here]. I'm [Describe your physical appearance here]. I'm [Describe your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, diverse culture, and vibrant nightlife. It is a major transportation hub, with the Paris Metro and the Eiffel Tower serving as major transportation arteries. The city is also home to the French Parliament and the French Academy of Sciences. Paris is a city of contrasts, with its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient AI systems that can better understand and respond to human needs.
    
    2. Enhanced privacy and security: As AI systems become more complex and sophisticated, there will be increased concerns about privacy and security. There will be a need for more robust privacy protections and security measures to ensure that AI systems are
    


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
    Generated text:  [Your Name], and I'm a writer, editor, and marketer. As an avid reader and longtime fan of [insert a favorite author], [Your Name] is passionate about helping writers improve their craft and selling more books. With over 5 years of experience in the industry, [Your Name] is dedicated to helping authors create impactful stories and building their brand. Looking to expand your knowledge and skills? If you're interested in writing, marketing, or both, [Your Name] is here to help. Contact me today and let's get started on your creative journey! [Your Name] [Your Email/Phone Number] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    The statement is: Paris is the capital of France.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by several key trends, including:
    
    1. Increased use of AI in healthcare: AI is being increasingly used to assist doctors in making diagnoses and treatment recommendations, improving patient outcomes, and reducing medical errors. This is likely to continue, with AI-powered diagnostic tools becoming more accurate and accessible to patients.
    
    2. AI in finance: AI is being used in a wide range of financial applications, from fraud detection and risk assessment to investment strategies and portfolio management. As AI technology improves, it is likely to become even more powerful, enabling even greater efficiency and accuracy in financial decision-making.
    
    3. AI in manufacturing: AI is being


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

    insert

     name

    ],

     and

     I

     am

     a

     [

    insert

     profession

    ].

     I

     am

     known

     for

     my

     [

    insert

     something

     interesting

     about

     yourself

    ].

     I

     enjoy

     [

    insert

     something

     interesting

     about

     yourself

    ].

     And

     what

     do

     you

     think

     makes

     me

     so

     great

    ?

     (

    offer

     a

     brief

     response

     or

     list

     of

     reasons

    )

     This

     is

     my

     brief

     introduction

    ,

     and

     I

     hope

     it

     gives

     you

     an

     idea

     of

     who

     I

     am

     as

     a

     character

    .

     It

    's

     always

     great

     to

     meet

     someone

     who

     is

     true

     to

     themselves

     and

     passionate

     about

     their

     craft

    .

     Have

     a

     great

     day

    !

     [

    insert

     self

    -int

    roduction

    ]

     [

    insert

     name

    ]

     is

     a

     skilled

     [

    insert

     profession

    ]

     who

     has

     a

     passion

     for

     [

    insert

     something

     interesting

     about

     themselves

    ].

     Despite

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     known

     for

     its

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

     E

    iff

    el

     Tower

    .

     The

     city

     is

     also

     a

     cultural

     and

     political

     center

    ,

     with

     a

     rich

     history

     dating

     back

     to

     the

     Middle

     Ages

     and

     the

     French

     Revolution

    .

     Its

     vibrant

     nightlife

     and

     traditional

     French

     cuisine

     contribute

     to

     its

     reputation

     as

     a

     popular

     tourist

     destination

    .

     Paris

     is

     known

     for

     its

     art

    ,

     music

    ,

     fashion

    ,

     and

     gastr

    onomy

    .

     Its

     French

     and

     French

     Cre

    ole

     communities

     also

     contribute

     to

     the

     city

    's

     diverse

     and

     colorful

     culture

    .

     Paris

     is

     an

     important

     international

     city

     with

     a

     large

     number

     of

     international

     organizations

     and

     conferences

    .

     It

     is

     also

     a

     major

     research

     and

     education

     center

    .

     The

     city

    's

     impressive

     skyline

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     characterized

     by

     a

     blend

     of

     currently

     un

    explo

    red

     and

     yet

     highly

     competitive

     areas

     of

     research

     and

     development

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     and

     social

     implications

    :

     With

     the

     increasing

     number

     of

     privacy

     concerns

     and

     the

     potential

     for

     AI

     to

     be

     used

     for

     malicious

     purposes

    ,

     there

     will

     be

     a

     greater

     emphasis

     on

     ethical

     and

     social

     implications

     of

     AI

    .

     This

     could

     lead

     to

     new

     technologies

     being

     developed

     with

     the

     aim

     of

     ensuring

     that

     AI

     is

     used

     for

     positive

     social

     outcomes

    .
    


    2

    .

     More

     collaboration

     between

     AI

     researchers

     and

     other

     fields

    :

     As

     AI

     becomes

     more

     integrated

     into

     everyday

     life

    ,

     it

     is

     likely

     to

     have

     a

     significant

     impact

     on

     other

     industries

     and

    



```python
llm.shutdown()
```
