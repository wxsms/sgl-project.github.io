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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.92it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.91it/s]


    2026-04-15 05:38:27,518 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-15 05:38:27] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.70s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.70s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.70s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:34,  2.70s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:02<00:22,  2.34it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:02<00:06,  6.81it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:02<00:06,  6.81it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:02<00:06,  6.81it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:02<00:06,  6.81it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:02<00:06,  6.81it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:02<00:06,  6.81it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:02<00:06,  6.81it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:02<00:06,  6.81it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:03<00:06,  6.81it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:03<00:06,  6.81it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:03<00:02, 13.91it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:03<00:01, 22.04it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:03<00:01, 22.04it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:03<00:01, 22.04it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:03<00:01, 22.04it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:03<00:01, 22.04it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:03<00:01, 22.04it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:03<00:01, 22.04it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:03<00:01, 22.04it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:03<00:01, 22.04it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:03<00:01, 22.04it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:03<00:00, 30.94it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:03<00:00, 30.94it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:03<00:00, 30.94it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:03<00:00, 30.94it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:03<00:00, 30.94it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:03<00:00, 30.94it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:03<00:00, 30.94it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:03<00:00, 30.94it/s]

    Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:03<00:00, 30.94it/s] Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:03<00:00, 30.94it/s]Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:03<00:00, 39.80it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:03<00:00, 39.80it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:03<00:00, 39.80it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:03<00:00, 39.80it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:03<00:00, 39.80it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:03<00:00, 39.80it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:03<00:00, 39.80it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:03<00:00, 39.80it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:03<00:00, 39.80it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:03<00:00, 39.80it/s] Compiling num tokens (num_tokens=4):  83%|████████▎ | 48/58 [00:03<00:00, 39.80it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 50.38it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=137.42 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=137.39 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=7168 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=6656 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   3%|▎         | 2/58 [00:00<00:02, 18.91it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=5632 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4608 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):   9%|▊         | 5/58 [00:00<00:02, 22.00it/s]Capturing num tokens (num_tokens=4096 avail_mem=137.38 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=3840 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=3584 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=3328 avail_mem=137.37 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.57it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.17it/s]Capturing num tokens (num_tokens=2560 avail_mem=137.36 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.17it/s]Capturing num tokens (num_tokens=2304 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.17it/s]Capturing num tokens (num_tokens=2048 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  22%|██▏       | 13/58 [00:00<00:01, 31.17it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  31%|███       | 18/58 [00:00<00:01, 35.40it/s]Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 35.40it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  31%|███       | 18/58 [00:00<00:01, 35.40it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  31%|███       | 18/58 [00:00<00:01, 35.40it/s]

    Capturing num tokens (num_tokens=960 avail_mem=137.31 GB):  31%|███       | 18/58 [00:00<00:01, 35.40it/s] Capturing num tokens (num_tokens=960 avail_mem=137.31 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=896 avail_mem=137.31 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=832 avail_mem=137.31 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=768 avail_mem=137.30 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=704 avail_mem=137.28 GB):  38%|███▊      | 22/58 [00:00<00:01, 31.41it/s]Capturing num tokens (num_tokens=704 avail_mem=137.28 GB):  45%|████▍     | 26/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=640 avail_mem=136.80 GB):  45%|████▍     | 26/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=576 avail_mem=136.80 GB):  45%|████▍     | 26/58 [00:00<00:01, 31.82it/s]

    Capturing num tokens (num_tokens=512 avail_mem=136.63 GB):  45%|████▍     | 26/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=480 avail_mem=136.65 GB):  45%|████▍     | 26/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=448 avail_mem=136.65 GB):  45%|████▍     | 26/58 [00:00<00:01, 31.82it/s]Capturing num tokens (num_tokens=448 avail_mem=136.65 GB):  53%|█████▎    | 31/58 [00:00<00:00, 35.22it/s]Capturing num tokens (num_tokens=416 avail_mem=136.64 GB):  53%|█████▎    | 31/58 [00:00<00:00, 35.22it/s]Capturing num tokens (num_tokens=384 avail_mem=136.64 GB):  53%|█████▎    | 31/58 [00:00<00:00, 35.22it/s]Capturing num tokens (num_tokens=352 avail_mem=136.64 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=320 avail_mem=136.63 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=288 avail_mem=136.63 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.22it/s]Capturing num tokens (num_tokens=288 avail_mem=136.63 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=256 avail_mem=136.63 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.60it/s]

    Capturing num tokens (num_tokens=240 avail_mem=136.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=224 avail_mem=136.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=208 avail_mem=136.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=192 avail_mem=136.62 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.60it/s]Capturing num tokens (num_tokens=192 avail_mem=136.62 GB):  71%|███████   | 41/58 [00:01<00:00, 38.92it/s]Capturing num tokens (num_tokens=176 avail_mem=136.61 GB):  71%|███████   | 41/58 [00:01<00:00, 38.92it/s]Capturing num tokens (num_tokens=160 avail_mem=136.61 GB):  71%|███████   | 41/58 [00:01<00:00, 38.92it/s]Capturing num tokens (num_tokens=144 avail_mem=136.61 GB):  71%|███████   | 41/58 [00:01<00:00, 38.92it/s]Capturing num tokens (num_tokens=128 avail_mem=136.60 GB):  71%|███████   | 41/58 [00:01<00:00, 38.92it/s]Capturing num tokens (num_tokens=112 avail_mem=136.60 GB):  71%|███████   | 41/58 [00:01<00:00, 38.92it/s]

    Capturing num tokens (num_tokens=112 avail_mem=136.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.22it/s]Capturing num tokens (num_tokens=96 avail_mem=136.60 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.22it/s] Capturing num tokens (num_tokens=80 avail_mem=136.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.22it/s]Capturing num tokens (num_tokens=64 avail_mem=136.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.22it/s]Capturing num tokens (num_tokens=48 avail_mem=136.59 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.22it/s]Capturing num tokens (num_tokens=32 avail_mem=136.58 GB):  79%|███████▉  | 46/58 [00:01<00:00, 40.22it/s]Capturing num tokens (num_tokens=32 avail_mem=136.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.93it/s]Capturing num tokens (num_tokens=28 avail_mem=136.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.93it/s]Capturing num tokens (num_tokens=24 avail_mem=136.58 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.93it/s]Capturing num tokens (num_tokens=20 avail_mem=136.57 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.93it/s]Capturing num tokens (num_tokens=16 avail_mem=136.57 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.93it/s]

    Capturing num tokens (num_tokens=12 avail_mem=136.57 GB):  88%|████████▊ | 51/58 [00:01<00:00, 40.93it/s]Capturing num tokens (num_tokens=12 avail_mem=136.57 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=8 avail_mem=136.56 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.97it/s] Capturing num tokens (num_tokens=4 avail_mem=136.56 GB):  97%|█████████▋| 56/58 [00:01<00:00, 41.97it/s]Capturing num tokens (num_tokens=4 avail_mem=136.56 GB): 100%|██████████| 58/58 [00:01<00:00, 36.34it/s]


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
    Generated text:  Erika and I am a PhD student in the School of Mathematics at the University of Birmingham.\nMy research interests lie in the intersection of geometric and harmonic analysis. I study the Fourier-Bessel transform of the Gaussian measure on the real line and its generalizations in higher dimensions, which I am interested in from the point of view of the theory of harmonic functions, through its applications to the analysis of random processes and the theory of random measures.\nIn my thesis I have studied the inverse problem of the Fourier transform and the inverse Fourier-Bessel transform, as well as an inverse problem related to the inverse Fourier-Bessel transform, based on
    ===============================
    Prompt: The president of the United States is
    Generated text:  a figure in the government of the United States who is elected every four years. In the United States, it is not uncommon for a president to be re-elected for consecutive terms. The president of the United States is also known as the chief executive of the United States.
    President of the United States
    The office of the President of the United States is a high-ranking post in the government of the United States. It is the highest ranking position within the executive branch of the U.S. government. The President of the United States is the head of the executive branch of the federal government and is the highest person in the government of the United States
    ===============================
    Prompt: The capital of France is
    Generated text:  ____
    A. Paris
    B. Tokyo
    C. Berlin
    D. Moscow
    Answer:
    A
    
    The 19th National Congress of the Communist Party of China pointed out that we should deepen educational system reform, build a quality-oriented education system, and improve the quality of education. Which of the following views about education is correct?
    A. Education is the fundamental solution to all problems.
    B. Education determines social progress and development.
    C. Education is the cornerstone of national rejuvenation and social progress.
    D. Education is the key to all affairs.
    Answer:
    C
    
    The most likely diagnosis for a patient with syncope is
    ===============================
    Prompt: The future of AI is
    Generated text:  a bright one
    
    This is one of the most exciting areas of technology in recent decades. It’s a field that, while still a relatively nascent one, has already achieved a number of breakthroughs in its ability to automate and augment existing human activities. Think about it – your phone can now write notes, learn a language, and play chess. Or your devices can now collaborate with the military to defend the nation against any cyber-attacks. Or your fitness tracker can now help you manage your expenses and plan your next trip. Or your smart home devices can now control your lights and thermostats to make your life easier. And the


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? [Name] is a [job title] at [company name]. I'm excited to meet you and learn more about
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major center for art, culture, and politics, and is home to many of the world's most famous museums and attractions. Paris is a vibrant and diverse city with a rich history and a strong sense of French identity. Its status as the capital of France has made it a major economic and cultural hub, and its influence extends beyond the city limits to other parts of the country. Paris is also known for its cuisine, with its famous dishes such as croissants, escarg
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in several key areas, including:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and efficient solutions to complex problems.
    
    2. Enhanced machine learning capabilities: AI is likely to become more powerful and capable, with the ability to learn from vast amounts of data and make more accurate predictions and decisions. This could lead to more efficient and effective decision-making in a wide range of applications.
    
    3. Increased use of AI in healthcare: AI is likely to play a more significant role
    


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
    Generated text:  Emily, I'm a strong-willed, independent woman with a passion for photography. I'm always looking for new and exciting subjects to capture, and I've been taking photos for years now. I'm always learning and growing, and I love sharing my knowledge with others. If you want to know more about me, please feel free to ask me anything, and I'll be happy to answer. Thank you. Hi, my name is Emily, I'm a strong-willed, independent woman with a passion for photography. I'm always looking for new and exciting subjects to capture, and I've been taking photos for years now. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France and serves as the seat of the government and the capital of France. It is located in the Île de la Cité, a historic city on the Seine River, about north of Paris, France. 
    
    Paris is renowned for its architecture, including the iconic Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral, as well as its diverse food scene and traditional French culture. It is also a cosmopolitan city with many international museums, music venues, and nightlife.
    
    Paris is often referred to as "The City of Light" due to its illuminated architecture and vibrant atmosphere
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to involve several key trends, including:
    
    1. Increased autonomy: AI systems will become more autonomous, able to make decisions and take action without human intervention. This will require greater levels of data collection and analysis, as well as more sophisticated algorithms and machine learning techniques.
    
    2. Semantic understanding: AI systems will become more capable of understanding natural language, recognizing context, and understanding the relationships between different entities and concepts. This will require greater levels of domain knowledge and expertise.
    
    3. Personalization: AI will become increasingly personalized, with the ability to adapt and optimize responses to individual users based on their behavior, preferences, and interests. This will


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

    ].

     I

     am

     a

     [

    type

     of

     character

    ],

     [

    character

    's

     role

     in

     the

     story

    ].

     I

     am

     the

     [

    role

    ]

     of

     [

    story

    line

    /

    setting

    ].

     I

     am

     [

    name

    ]

     and

     I

     am

     from

     [

    city

    ,

     country

    ,

     state

    ,

     etc

    .

    ].

     I

     am

     a

     [

    occupation

    ,

     hobby

    ,

     etc

    .

    ].

     Here

    's

     what

     you

     can

     expect

     from

     me

     in

     this

     story

    :
    


    As

     the

     [

    type

     of

     character

    ],

     [

    character

    's

     role

     in

     the

     story

    ],

     I

     am

     a

     character

     who

     will

     [

    describe

     the

     action

     or

     ability

    ].

     I

     am

     a

     [

    describe

     the

     character

    's

     personality

    ]

     and

     I

     am

     always

     [

    describe

     the

     character

    's

     traits

    ].
    


    I

     am

     a

     [

    describe

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     home

     to

     the

     iconic

     E

    iff

    el

     Tower

    .

     Known

     as

     "

    City

     of

     Light

    ",

     Paris

     is

     an

    国际化

    的大

    都市

    （

    国际化

    大

    都市

    ）

    ,

     with

     a

     rich

     cultural

     and

     artistic

     heritage

    .

     The

     city

     is

     also

     a

     popular

     tourist

     destination

    ,

     known

     for

     its

     stunning

     architecture

    ,

     vibrant

     nightlife

    ,

     and

     world

    -class

     museums

     and

     museums

    .

     With

     a

     population

     of

     over

     a

     million

     people

    ,

     Paris

     is

     a

     bustling

     met

    ropolis

     that

     has

     played

     a

     significant

     role

     in

     French

     history

     and

     culture

    .

     As

     of

     

    2

    0

    2

    1

    ,

     the

     city

     is

     home

     to

     the

     European

     Capital

     of

     Culture

    ,

     hosting

     a

     variety

     of

     cultural

     events

     and

     activities

     throughout

     the

     year

    .

     Additionally

    ,

     the

     French

     government

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

    ,

     and

     here

     are

     some

     of

     the

     possible

     trends

    :
    


    1

    .

     Increased

     automation

     and

     productivity

    :

     AI

     is

     already

     revolution

    izing

     many

     industries

    ,

     and

     we

     can

     expect

     it

     to

     become

     even

     more

     pervasive

     in

     the

     coming

     years

    .

     Automation

     is

     expected

     to

     increase

     efficiency

    ,

     reduce

     costs

    ,

     and

     increase

     productivity

    .
    


    2

    .

     AI

     in

     healthcare

    :

     AI

     is

     being

     used

     to

     develop

     new

     diagnostic

     tools

    ,

     treatments

    ,

     and

     therapies

    ,

     which

     are

     expected

     to

     improve

     patient

     outcomes

    .

     AI

     is

     also

     being

     used

     to

     develop

     more

     accurate

     and

     personalized

     medical

     treatments

    .
    


    3

    .

     AI

     in

     education

    :

     AI

     is

     being

     used

     to

     personalize

     learning

     experiences

    ,

     adjust

     teaching

     methods

    ,

     and

     improve

     academic

     performance

    .

     AI

    -powered

     educational

     tools

     are

     expected

    



```python
llm.shutdown()
```
