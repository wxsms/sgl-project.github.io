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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.23it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.23it/s]


    2026-05-17 22:04:57,696 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-17 22:04:57] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:58,  4.18s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.55it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.63it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.63it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.35it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.60it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.60it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.60it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.60it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.60it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.60it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.60it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.60it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.60it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.60it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.60it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.01it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.26it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.26it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.26it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.26it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.26it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.26it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.26it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.26it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.94it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.20 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.16 GB):   3%|▎         | 2/58 [00:00<00:02, 19.24it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.50it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.50it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.50it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.50it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.40it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.40it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.40it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.40it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.11 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.11 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.11it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.08it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.08it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.08it/s]Capturing num tokens (num_tokens=960 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.08it/s] Capturing num tokens (num_tokens=896 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.08it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.08it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=768 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=704 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=512 avail_mem=74.07 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.46it/s]Capturing num tokens (num_tokens=512 avail_mem=74.07 GB):  50%|█████     | 29/58 [00:00<00:00, 43.80it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 43.80it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  50%|█████     | 29/58 [00:00<00:00, 43.80it/s]Capturing num tokens (num_tokens=416 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:00<00:00, 43.80it/s]Capturing num tokens (num_tokens=384 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:00<00:00, 43.80it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  50%|█████     | 29/58 [00:00<00:00, 43.80it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.43it/s]Capturing num tokens (num_tokens=320 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.43it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.43it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.43it/s]Capturing num tokens (num_tokens=240 avail_mem=74.06 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.43it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.43it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.57it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.57it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.57it/s]Capturing num tokens (num_tokens=176 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.57it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.57it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.57it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=128 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=112 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.11it/s] Capturing num tokens (num_tokens=80 avail_mem=74.03 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.11it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  76%|███████▌  | 44/58 [00:01<00:00, 46.11it/s]

    Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=32 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=20 avail_mem=74.01 GB):  84%|████████▍ | 49/58 [00:01<00:00, 37.12it/s]Capturing num tokens (num_tokens=20 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=16 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  93%|█████████▎| 54/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=8 avail_mem=74.00 GB):  93%|█████████▎| 54/58 [00:01<00:00, 37.05it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  93%|█████████▎| 54/58 [00:01<00:00, 37.05it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 37.39it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 38.06it/s]


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
    Generated text:  [Name], and I am a [Type] [Age] year old [Gender] [Color]. I am an [Occupation] [Job] and I have a lot of [Attractions] [Activities]. I have [Number of Pets] [Pets]. I would like to ask you a question about yourself. Can you tell me about yourself? 
    
    My name is [Name], and I am a [Type] [Age] year old [Gender] [Color]. I am an [Occupation] [Job] and I have a lot of [Attractions] [Activities]. I have [Number of Pets] [
    ===============================
    Prompt: The president of the United States is
    Generated text:  considering a policy to reduce carbon emissions by imposing a tax on carbon emissions. The tax is to be applied to carbon emissions from both coal-fired power plants and natural gas-fired power plants. Coal-fired power plants emit 100 tons of carbon dioxide (CO2) per year, and natural gas-fired power plants emit 80 tons of CO2 per year. The tax is to be proportional to the emissions, and the amount of tax collected must be at least $500,000.
    
    Assuming that the tax rate for coal-fired power plants is $20 per ton and for natural gas-fired power plants is
    ===============================
    Prompt: The capital of France is
    Generated text:  _____.
    A. Paris
    B. Lille
    C. Marseille
    D. Lyon
    
    The capital of France is Paris. Therefore, the correct answer is A. Paris. 
    
    The other options are not capital cities of France: Lille is in northern France, Marseille is in south France, and Lyon is in northeastern France. Paris is the largest city in France by population and is located in the north-central region of the country. It is known for its landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.
    ===============================
    Prompt: The future of AI is
    Generated text:  highly uncertain. While there are many potential benefits and improvements to be seen, there is also a growing risk of unintended consequences and negative impacts on society. One of the most pressing concerns is the potential for AI to be used to perpetuate existing inequalities and biases. AI systems can be designed and deployed to perpetuate existing power structures, with disparities in access and participation likely to persist.
    The impact of AI on marginalized communities is often further compounded by the systemic inequalities they face, such as racism, sexism, homophobia, and ableism. AI systems can be designed and deployed to exacerbate these inequalities and create further barriers to marginalized communities, while


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your profession or role]. I'm always looking for new challenges and opportunities to grow and learn. What do you do for a living? I'm always looking for new ways to improve myself and help others. What's your favorite hobby or activity? I enjoy reading, playing sports, and spending time with my family. What's your favorite book or movie? I love [insert a favorite book or movie]. What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also known for its fashion industry, with many famous fashion designers and boutiques located in the city. The city is also home to the French Riviera, a popular tourist destination for its beaches and luxury resorts. Overall, Paris is a vibrant and diverse city with a rich history and culture.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such as machine learning, natural language processing, and computer vision. This will enable AI to perform a wider range of tasks and improve its performance.
    
    3. Development of new
    


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
    Generated text:  [Name]. I'm a [Occupation] with [Experience Level]. I am a professional [Job Title], and I have [Number of Years in the Job]. I am dedicated to [Your Profession]. I am passionate about [My Passion or Hobby]. I believe in [My Core Values or Principles]. I have a [Number of Years in This Profession]. I am [Your Interests, such as reading, travel, or music], and I enjoy [My Hobby or Interests]. I am [Your Personality Type or Personality]. I have always been [Your Qualifications or Qualifications], and I am [Your Knowledge or Knowledge
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower and distinctive style of architecture. It's a bustling city with a rich cultural history, including the Louvre Museum, Notre-Dame Cathedral, and the Palace of Versailles. Paris is also famous for its annual时尚节，如La Fête des Glaces and the Biennale de Arte Moderno. As a major world city, Paris offers many attractions and cultural events for visitors. Its complex infrastructure, including metro and bus systems, makes it a convenient city for both locals and tourists. The French government supports efforts to preserve and improve Paris, as evidenced by the ongoing renovation of the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by a growing number of applications, increasing complexity, and a focus on ethical considerations. Here are some possible trends that may emerge in the AI landscape in the coming years:
    
    1. Increased accessibility: As AI technology continues to improve, we may see more accessible AI applications for everyday use, such as voice assistants, self-driving cars, and virtual assistants. This could make AI a more ubiquitous and accessible technology for people of all ages and backgrounds.
    
    2. More sophisticated models: As AI technology advances, we may see more sophisticated models that can learn from large amounts of data and make more accurate predictions and decisions. This could lead


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

    ]

     and

     I

    'm

     a

     [

    Your

     Profession

    /

    Position

    ]

     with

     [

    Your

     Key

     Skills

    /

    Experience

    ]

     in

     [

    Your

     Key

     Area

     of

     Expert

    ise

    ].

     I

    'm

     here

     to

     share

     my

     knowledge

     and

     experiences

     with

     you

     as

     a

     [

    Your

     Inter

    ests

    /

    Goals

    ]

     and

     contribute

     to

     the

     knowledge

     base

    .

     What

     can

     you

     tell

     me

     about

     yourself

     and

     your

     area

     of

     expertise

    ?

     [

    Your

     Name

    ]

     is

     a

     dedicated

     and

     passionate

     expert

     in

     [

    Your

     Field

    /

    Industry

    ],

     known

     for

     my

     [

    Your

     Key

     Skills

    /

    Experience

    ]

     that

     have

     helped

     me

     excel

     in

     my

     career

    .

     I

     am

     always

     eager

     to

     learn

     and

     improve

    ,

     and

     I

    'm

     always

     happy

     to

     share

     my

     knowledge

     with

     anyone

     who

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     world

    -ren

    owned

     cultural

    ,

     historical

    ,

     and

     economic

     center

    .


    Paris

    ,

     known

     as

     "

    the

     City

     of

     Light

    "

     and

     "

    the

     City

     of

     Love

    ,"

     is

     the

     largest

     city

     in

     France

     and

     the

     fourth

    -largest

     city

     in

     the

     world

    .

     It

     is

     located

     in

     the

     south

     of

     France

    ,

     on

     the

     banks

     of

     the

     Se

    ine

     River

    ,

     on

     the

     outskirts

     of

     the

     Paris

     Basin

    .

     The

     city

     is

     home

     to

     millions

     of

     people

    ,

     and

     its

     cultural

     and

     economic

     landscape

     is

     diverse

    ,

     featuring

     architecture

    ,

     museums

    ,

     restaurants

    ,

     and

     nightlife

    .

     Paris

     is

     the

     gateway

     to

     France

     and

     the

     world

    ,

     attracting

     millions

     of

     tourists

     annually

     and

     becoming

     an

     important

     economic

     hub

     for

     the

     country

     and

     beyond

    .

     While

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     diverse

     and

     unpredictable

    ,

     with

     many

     potential

     developments

     shaping

     the

     technology

    's

     direction

     and

     impact

    .

     Some

     potential

     trends

     include

    :
    


    1

    .

     Improved

     algorithms

     and

     models

    :

     As

     AI

     continues

     to

     advance

    ,

     we

     may

     see

     more

     advanced

     algorithms

     and

     models

     that

     can

     better

     understand

     and

     analyze

     complex

     data

     sets

    .

     This

     could

     lead

     to

     more

     accurate

     predictions

     and

     recommendations

    ,

     as

     well

     as

     the

     ability

     to

     solve

     complex

     problems

     that

     were

     previously

     beyond

     the

     reach

     of

     AI

    .
    


    2

    .

     Increased

     integration

     with

     other

     technologies

    :

     AI

     is

     becoming

     increasingly

     integrated

     into

     other

     technologies

    ,

     such

     as

     smart

     homes

     and

     transportation

     systems

    ,

     which

     could

     lead

     to

     even

     more

     seamless

     and

     efficient

     interactions

     between

     people

     and

     machines

    .

     This

     could

     also

     increase

     the

    



```python
llm.shutdown()
```
