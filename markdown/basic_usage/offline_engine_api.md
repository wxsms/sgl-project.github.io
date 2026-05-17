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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.12it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.11it/s]


    2026-05-17 15:39:13,210 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 15:39:13] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:24,  4.63s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:37,  1.40it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.20it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.20it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.44it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:03,  9.44it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:03,  9.44it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:03,  9.44it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:03,  9.44it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 15.17it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 15.17it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 15.17it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 15.17it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 15.17it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 15.17it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 15.17it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 15.17it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 15.17it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 15.17it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 15.17it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 23.02it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 23.02it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 23.02it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 23.02it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 23.02it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 23.02it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 23.02it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 23.02it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 23.02it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.02it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.02it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 31.64it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 31.64it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 31.64it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 31.64it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 31.64it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 31.64it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 31.64it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 31.64it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.87it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.66 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.63 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.62 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.61 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):   9%|▊         | 5/58 [00:00<00:02, 22.17it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=3840 avail_mem=72.60 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.59 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.98it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.98it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.59it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=72.58 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.59it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.59it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.59it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.57 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=960 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s] Capturing num tokens (num_tokens=896 avail_mem=72.56 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]

    Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.37it/s]Capturing num tokens (num_tokens=832 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=768 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=704 avail_mem=72.55 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=640 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=576 avail_mem=72.54 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.79it/s]Capturing num tokens (num_tokens=512 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=480 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=448 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=416 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=384 avail_mem=72.54 GB):  50%|█████     | 29/58 [00:00<00:00, 42.92it/s]

    Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  50%|█████     | 29/58 [00:00<00:00, 42.92it/s]Capturing num tokens (num_tokens=352 avail_mem=72.53 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.80it/s]Capturing num tokens (num_tokens=320 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.80it/s]Capturing num tokens (num_tokens=288 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.80it/s]Capturing num tokens (num_tokens=256 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.80it/s]Capturing num tokens (num_tokens=240 avail_mem=72.52 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.80it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  59%|█████▊    | 34/58 [00:00<00:00, 44.80it/s]Capturing num tokens (num_tokens=224 avail_mem=72.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=208 avail_mem=72.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=192 avail_mem=72.23 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=176 avail_mem=72.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.90it/s]

    Capturing num tokens (num_tokens=160 avail_mem=72.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  67%|██████▋   | 39/58 [00:01<00:00, 42.90it/s]Capturing num tokens (num_tokens=144 avail_mem=72.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=128 avail_mem=72.22 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=112 avail_mem=72.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=96 avail_mem=72.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.65it/s] Capturing num tokens (num_tokens=80 avail_mem=72.21 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.65it/s]Capturing num tokens (num_tokens=64 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=48 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=32 avail_mem=72.20 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=28 avail_mem=72.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.80it/s]

    Capturing num tokens (num_tokens=24 avail_mem=72.19 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=20 avail_mem=72.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.80it/s]Capturing num tokens (num_tokens=16 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=12 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=8 avail_mem=72.18 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.19it/s] Capturing num tokens (num_tokens=4 avail_mem=72.17 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.19it/s]Capturing num tokens (num_tokens=4 avail_mem=72.17 GB): 100%|██████████| 58/58 [00:01<00:00, 41.28it/s]


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
    Generated text:  Jolie and I'm a graphic designer at Harmony. I specialize in design in the visual communication fields. My focus areas are communication design, copywriting, and logo design. I'm an avid learner and always looking for new ways to improve my skills and knowledge in my field.
    What is communication design?
    Communication design is the study of how we create, communicate, and interpret messages and messages to audiences and individuals. This includes the application of graphic design principles, principles of human-computer interaction, and user-centered design, as well as the application of design to a wide range of contexts, including but not limited to, marketing, public relations
    ===============================
    Prompt: The president of the United States is
    Generated text:  at the presidency house on the west side of Washington D. C. , from where he has long been considered the greatest American hero, admired by the citizens of America, and feared by those who oppose the government. He has been given many honors and decorations, including the 1995 Presidential Medal of Freedom, the highest award for service to the United States. He has been married three times and has seven children, the eldest of whom is the son of former president George H. W. Bush. At the time of his death, he was the owner of two of the world's largest corporations - Johnson & Johnson and Merck
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It is the largest city in Europe and one of the most populous cities in the world. It is situated on the banks of the river Seine, which runs through the heart of the city and connects the Seine Valley and the Seine-Marne Valley.
    The Seine is 150 km long and it is 500 m wide, while the main bridge is 125 m wide and 516 m long. Its width varies depending on the direction. The width of the Seine at its highest point is 528 m.
    The Seine takes over 25 years to
    ===============================
    Prompt: The future of AI is
    Generated text:  poised for significant growth, but it will also face some challenges. AI will continue to be used in new ways and new areas of our lives, but we must also consider how it will affect people and society as a whole. The following are some of the challenges that AI will face in the future:
    1. Bias and discrimination: One of the most significant challenges that AI will face in the future is the potential for bias and discrimination in its algorithms and decision-making processes. AI systems are only as good as the data they are trained on, and if that data contains biases or is biased, the AI will be biased as well. This can


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


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and a diverse population of over 2 million people. It is the largest city in Europe by population and is a major economic and political center for France. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a popular tourist destination, with millions of visitors each year.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As the technology continues to advance, we can expect to see even more widespread use of AI in healthcare, including in areas such as diagnosis, treatment planning, and patient monitoring.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve fraud detection, risk assessment, and investment decision-making.
    


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
    Generated text:  [Name] and I'm [Age] years old. I'm a [job title] at [company name] who specializes in [what you do]. I've always been fascinated by [curiosity or interest] and have always wanted to [specific goal or achievement]. What's your favorite [current hobby, sport, food, etc.]? I'm always looking for ways to [something I'm passionate about or interested in]. I've been [number of years] with [job title] at [company name] and I enjoy [what you enjoy doing]. I'm really excited to meet you and see what you have to offer
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    The statement is: "Paris is the capital city of France." This concise statement encapsulates the central fact about Paris's role as the nation's capital and exemplifies how it fits within the broader context of French political and administrative structures. While not providing any additional information beyond this core statement, it effectively communicates the essential information about Paris's status as the capital. 
    
    To elaborate further, Paris is France's largest metropolitan area, covering the metropolitan region between Paris, the Seine River, and the Rhône Valley. It is home to numerous cultural, educational, and international institutions, including the Louvre Museum, the Palace of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising and will continue to evolve in numerous ways. Here are some possible trends in the coming years:
    
    1. Increased integration with other technologies: AI will continue to integrate more deeply with other technologies such as the internet, sensors, and blockchain, leading to a more integrated and connected world.
    
    2. Better accuracy and efficiency: AI will continue to improve its ability to perform tasks with greater accuracy and efficiency, making it more accessible and useful for a wide range of applications.
    
    3. Personalization and adaptability: AI will continue to become more personalized and adaptable, allowing machines to learn from data and improve their performance over time.
    
    4. Better


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

     __

    ________

    _

     and

     I

    'm

     a

    (n

    )

     ______

    __

    _.

     I

     have

     a

     ______

    __

     experience

     in

     the

     field

     of

     ______

    __.

     I

    'm

     a

    (n

    )

     ______

    __

     who

     am

     ______

    __.

     I

    'm

     always

     ______

    __.

     I

     enjoy

     ______

    __.

     I

    'm

     a

    (n

    )

     ______

    __

     who

     is

     always

     ______

    __.

     I

    'm

     always

     ______

    __.

     I

    'm

     always

     ______

    __.

     I

    'm

     always

     ______

    __.

     I

    'm

     always

     ______

    __.

     


    [

    Your

     name

    ]

     is

     a

    (n

    )

     ______

    __

     who

     is

     ______

    __.

     I

    'm

     always

     ______

    __.

     I

     enjoy

     ______

    __.

     I

    'm

     always

     ______

    __.

     I

    'm

     always

     ______

    __.

     I

    'm

     always

     ______

    __.

     I

    'm

     always

     ______

    __.

     I

    'm

     always

     ______

    __.

     I

    'm

     always

    
    
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

     capital

     of

     France

    ,

     located

     on

     the

     French

     Riv

    iera

     on

     the

     Mediterranean

     Sea

    ,

     and

     is

     the

     most

     populous

     city

     of

     France

    .

     The

     city

     is

     also

     a

     major

     financial

     center

    ,

     a

     major

     hub

     of

     fashion

    ,

     and

     a

     hub

     of

     culture

     and

     education

    .

     Paris

     is

     home

     to

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

    ,

     as

     well

     as

     many

     other

     iconic

     landmarks

    .

     The

     city

     is

     also

     known

     for

     its

     cuisine

    ,

     art

    ,

     and

     history

    ,

     and

     is

     a

     popular

     tourist

     destination

     worldwide

    .

     Paris

     is

     often

     referred

     to

     as

     "

    La

     Ville

     Fl

    ipp

    ante

    ,"

     meaning

     "

    The

     Fl

    ipp

    ant

     Capital

    "

     in

     French

    .

     According

     to

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     bright

     and

     promising

     one

    ,

     with

     many

     exciting

     developments

     and

     advancements

     on

     the

     horizon

    .

     Here

     are

     some

     possible

     future

     trends

     in

     artificial

     intelligence

    :
    


    1

    .

     Increased

     integration

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     sophisticated

     and

     integrated

     with

     other

     technologies

    ,

     including

     sensors

    ,

     data

     analytics

    ,

     and

     blockchain

    ,

     we

     can

     expect

     to

     see

     more

     seamless

     and

     seamless

     integration

     between

     AI

     systems

     and

     other

     technologies

    .

     This

     will

     make

     it

     easier

     for

     businesses

     to

     collect

     and

     analyze

     data

     in

     real

    -time

    ,

     enabling

     more

     precise

     and

     informed

     decision

    -making

    .
    


    2

    .

     Autonomous

     vehicles

    :

     In

     the

     coming

     decades

    ,

     we

     can

     expect

     to

     see

     more

     autonomous

     vehicles

     on

     the

     roads

    ,

     with

     AI

    -powered

     self

    -driving

     cars

     becoming

     increasingly

    



```python
llm.shutdown()
```
