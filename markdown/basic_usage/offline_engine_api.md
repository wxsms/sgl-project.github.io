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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.90it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]


    2026-05-05 01:16:45,755 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-05 01:16:45] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.53it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.55it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.55it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.20it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=176):  53%|█████▎    | 31/58 [00:04<00:01, 16.39it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:04<00:00, 25.57it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=48):  72%|███████▏  | 42/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=32):  72%|███████▏  | 42/58 [00:04<00:00, 25.57it/s]

    Compiling num tokens (num_tokens=28):  72%|███████▏  | 42/58 [00:04<00:00, 25.57it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:04<00:00, 34.72it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:04<00:00, 34.72it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:04<00:00, 34.72it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:04<00:00, 34.72it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:04<00:00, 34.72it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:04<00:00, 34.72it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:04<00:00, 34.72it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.78it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=73.53 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.50 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=73.50 GB):   3%|▎         | 2/58 [00:00<00:03, 17.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.52 GB):   3%|▎         | 2/58 [00:00<00:03, 17.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.44 GB):   3%|▎         | 2/58 [00:00<00:03, 17.32it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=58.48 GB):   3%|▎         | 2/58 [00:00<00:03, 17.32it/s]Capturing num tokens (num_tokens=6144 avail_mem=58.48 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=5632 avail_mem=58.47 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=58.46 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=4608 avail_mem=58.46 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.46 GB):   9%|▊         | 5/58 [00:00<00:02, 20.92it/s]Capturing num tokens (num_tokens=4096 avail_mem=58.46 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]Capturing num tokens (num_tokens=3840 avail_mem=58.45 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]Capturing num tokens (num_tokens=3584 avail_mem=58.45 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=58.45 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]Capturing num tokens (num_tokens=3072 avail_mem=58.44 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.44 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.40it/s]Capturing num tokens (num_tokens=2816 avail_mem=58.44 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.53it/s]Capturing num tokens (num_tokens=2560 avail_mem=58.44 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.53it/s]Capturing num tokens (num_tokens=2304 avail_mem=58.43 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.53it/s]Capturing num tokens (num_tokens=2048 avail_mem=58.43 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.53it/s]Capturing num tokens (num_tokens=1792 avail_mem=58.43 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.42 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.53it/s]Capturing num tokens (num_tokens=1536 avail_mem=58.42 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.87it/s]Capturing num tokens (num_tokens=1280 avail_mem=58.42 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=58.40 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.87it/s]

    Capturing num tokens (num_tokens=960 avail_mem=58.42 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.87it/s] Capturing num tokens (num_tokens=896 avail_mem=58.41 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.87it/s]Capturing num tokens (num_tokens=832 avail_mem=58.41 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.87it/s]Capturing num tokens (num_tokens=832 avail_mem=58.41 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=768 avail_mem=58.41 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=704 avail_mem=58.40 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=640 avail_mem=58.40 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=576 avail_mem=58.40 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=512 avail_mem=58.39 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.33it/s]Capturing num tokens (num_tokens=512 avail_mem=58.39 GB):  50%|█████     | 29/58 [00:00<00:00, 41.04it/s]Capturing num tokens (num_tokens=480 avail_mem=58.40 GB):  50%|█████     | 29/58 [00:00<00:00, 41.04it/s]

    Capturing num tokens (num_tokens=448 avail_mem=58.40 GB):  50%|█████     | 29/58 [00:00<00:00, 41.04it/s]Capturing num tokens (num_tokens=416 avail_mem=58.40 GB):  50%|█████     | 29/58 [00:00<00:00, 41.04it/s]Capturing num tokens (num_tokens=384 avail_mem=58.40 GB):  50%|█████     | 29/58 [00:00<00:00, 41.04it/s]Capturing num tokens (num_tokens=352 avail_mem=58.39 GB):  50%|█████     | 29/58 [00:00<00:00, 41.04it/s]Capturing num tokens (num_tokens=352 avail_mem=58.39 GB):  59%|█████▊    | 34/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=320 avail_mem=58.38 GB):  59%|█████▊    | 34/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=288 avail_mem=58.38 GB):  59%|█████▊    | 34/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=256 avail_mem=58.38 GB):  59%|█████▊    | 34/58 [00:00<00:00, 39.78it/s]Capturing num tokens (num_tokens=240 avail_mem=58.38 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.78it/s]

    Capturing num tokens (num_tokens=224 avail_mem=58.37 GB):  59%|█████▊    | 34/58 [00:01<00:00, 39.78it/s]Capturing num tokens (num_tokens=224 avail_mem=58.37 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.08it/s]Capturing num tokens (num_tokens=208 avail_mem=58.37 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.08it/s]Capturing num tokens (num_tokens=192 avail_mem=58.37 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.08it/s]Capturing num tokens (num_tokens=176 avail_mem=58.36 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.08it/s]Capturing num tokens (num_tokens=160 avail_mem=58.36 GB):  67%|██████▋   | 39/58 [00:01<00:00, 39.08it/s]Capturing num tokens (num_tokens=160 avail_mem=58.36 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.78it/s]Capturing num tokens (num_tokens=144 avail_mem=58.36 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.78it/s]Capturing num tokens (num_tokens=128 avail_mem=58.36 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.78it/s]Capturing num tokens (num_tokens=112 avail_mem=58.35 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.78it/s]

    Capturing num tokens (num_tokens=96 avail_mem=58.35 GB):  74%|███████▍  | 43/58 [00:01<00:00, 38.78it/s] Capturing num tokens (num_tokens=96 avail_mem=58.35 GB):  81%|████████  | 47/58 [00:01<00:00, 38.47it/s]Capturing num tokens (num_tokens=80 avail_mem=58.35 GB):  81%|████████  | 47/58 [00:01<00:00, 38.47it/s]Capturing num tokens (num_tokens=64 avail_mem=58.34 GB):  81%|████████  | 47/58 [00:01<00:00, 38.47it/s]Capturing num tokens (num_tokens=48 avail_mem=58.34 GB):  81%|████████  | 47/58 [00:01<00:00, 38.47it/s]Capturing num tokens (num_tokens=32 avail_mem=58.34 GB):  81%|████████  | 47/58 [00:01<00:00, 38.47it/s]Capturing num tokens (num_tokens=32 avail_mem=58.34 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=28 avail_mem=58.33 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=24 avail_mem=58.33 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=20 avail_mem=58.32 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.12it/s]

    Capturing num tokens (num_tokens=16 avail_mem=58.32 GB):  88%|████████▊ | 51/58 [00:01<00:00, 38.12it/s]Capturing num tokens (num_tokens=16 avail_mem=58.32 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=12 avail_mem=58.32 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=8 avail_mem=58.32 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.22it/s] Capturing num tokens (num_tokens=4 avail_mem=58.31 GB):  95%|█████████▍| 55/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=4 avail_mem=58.31 GB): 100%|██████████| 58/58 [00:01<00:00, 36.81it/s]


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
    Generated text:  Danny. I'm from Peru. I came to the United States to study the natural resources of Peru. I'm a big fan of the Amazon. So I decided to visit the Amazon in Peru. When I came to Peru, the Amazon was in a state of drought. My first stop was an Amazonia park near the city of Cusco. It is a very beautiful place. I was able to see many different kinds of animals there. It was a very nice place to stay. The next stop was the National Museum. The museum is in the city of Cusco. It was not as beautiful as the park. I
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military bases to build. The cost of building the x bases is given by the function c(x) = 12x + 20000. If the cost of building one base is constant, how many bases should the president build to minimize the cost of the project? To minimize the cost of building the bases, the president needs to find the minimum value of the cost function \( c(x) = 12x + 20000 \). The cost function \( c(x) \) is a linear function, and the minimum value of a linear function occurs at the point
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 正确
    B. 错误
    Answer:
    
    A
    
    A university is planning to use a machine learning algorithm to predict the likelihood of students graduating from their courses. The algorithm is currently trained on a dataset of 100 students, and the training set is clean and has no missing values. However, the algorithm has an external error rate of 5%. If the algorithm is used to predict the likelihood of a student graduating, what is the minimum number of students required to ensure a 95% accuracy rate on the testing dataset, assuming the testing dataset also has no missing values and is entirely composed
    ===============================
    Prompt: The future of AI is
    Generated text:  in the hands of the individual.
    Yes, that’s the view of a group of researchers from Harvard University, who have published a paper in the journal Science that suggest that, if we want to achieve more than 50% of the world’s population on AI by 2050, we will need to use it for the benefit of the entire human race.
    So far, the world has seen 70% of AI being used in industrialised economies and just 10% in developing ones. However, the researchers have found that AI is the new industrial machinery that will be essential to the future economic development.
    In


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm excited to meet you and learn more about you. What
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville-Marie" or "La Ville de Paris". It is the largest city in France and the second-largest city in the European Union. Paris is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, with a rich history dating back to the Roman Empire. Paris is known for its vibrant nightlife, art scene, and delicious cuisine. It is a popular tourist destination and a major hub for business and commerce in France. The city is also home to many international organizations and institutions, including the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and privacy.
    
    2. Greater integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior. This will enable machines to perform tasks that are difficult or impossible for humans to do.
    
    3. Increased use of machine learning: Machine learning
    


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
    Generated text:  Sarah, I'm a writer, and I write science fiction. I've always been fascinated by the universe and the stars. My favorite author is Robert Jordan, and I love learning about the universe from his books. I write because I love to imagine what the future holds, and science fiction is a great way to explore that. What's your favorite book or movie that has inspired you? That would be really interesting to know. Hi, I'm Sarah, a science fiction writer. I love writing science fiction because it allows me to explore the stars, the future, and the unknown. I also really like reading and learning about science fiction
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic Eiffel Tower and the Louvre Museum. The city is also famous for its architecture, particularly the Baroque style, and its historical importance as a major city and political center. Paris is home to the headquarters of many major French companies and is a popular tourist destination, with its charming cafes, museums, and theaters. The French capital has a rich cultural history dating back to the Middle Ages and is home to numerous cultural institutions, including the Louvre, the National Museum of China, and the Champs-Élysées. It is considered one of the most beautiful and expensive cities in the world and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be one of rapid growth and development, driven by advancements in hardware and software. Here are some possible future trends in AI:
    
    1. Increased use of AI in healthcare: AI is being used to improve the accuracy and efficiency of medical diagnosis and treatment. For example, AI-powered medical devices can analyze medical images, interpret medical data, and recommend treatment plans. This could lead to more accurate diagnoses, earlier detection of diseases, and more effective treatments for chronic conditions.
    
    2. Autonomous vehicles: Autonomous vehicles are becoming increasingly common, and AI is playing a key role in their development. AI algorithms are being used to optimize driving routes, detect


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

     ____

     and

     I

    'm

     a

    /an

     ____

    _.

     I

    'm

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

     I

    'm

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

     I

    'm

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

     I

    'm

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

     I

    'm

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

     I

    'm

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

     I

    'm

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

     I

    'm

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

     I

    'm

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

     I

    'm

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

     I

    'm

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

     I

    'm

     a

    /an

     ____

    _.

     I

     enjoy

     ____

    _.

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Here

    's

     a

     concise

     factual

     statement

     about

     Paris

    :
    


    Paris

     is

     the

     capital

     and

     largest

     city

     of

     France

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

     south

    -central

     region

     of

     France

    .

     The

     city

     is

     known

     for

     its

     historic

     buildings

    ,

     cultural

     attractions

    ,

     and

     diverse

     cuisine

    .

     It

     is

     the

     third

     most

     populous

     city

     in

     France

     and

     is

     a

     major

     center

     of

     education

    ,

     media

    ,

     and

     business

    .

     Paris

     is

     often

     referred

     to

     as

     the

     "

    City

     of

     Light

    "

     and

     the

     "

    City

     of

     Light

    "

     for

     its

     iconic

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

     Notre

    -D

    ame

     Cathedral

    ,

     and

     Lou

    vre

     Museum

    ,

     among

     others

    .

     The

     city

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     fascinating

     and

     constantly

     evolving

    ,

     with

     potential

     changes

     in

     technology

    ,

     ethics

    ,

     and

     societal

     implications

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

     human

     capabilities

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

     capabilities

    ,

     particularly

     in

     areas

     such

     as

     automation

    ,

     perception

    ,

     and

     decision

    -making

    .

     AI

    -powered

     systems

     will

     be

     able

     to

     perform

     tasks

     that

     are

     currently

     the

     domain

     of

     humans

    ,

     such

     as

     medical

     diagnosis

    ,

     finance

    ,

     and

     transportation

    .
    


    2

    .

     Personal

    ization

     and

     adapt

    ability

    :

     AI

     will

     become

     more

     personal

    ,

     able

     to

     adapt

     to

     individual

     preferences

     and

     situations

    .

     For

     example

    ,

     AI

    -powered

     chat

    bots

     will

     be

     able

     to

     respond

     to

     questions

     in

     a

     natural

     and

     human

    



```python
llm.shutdown()
```
