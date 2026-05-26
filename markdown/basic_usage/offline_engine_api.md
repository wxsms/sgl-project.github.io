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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.58it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:02,  4.25s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:34,  1.52it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.53it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.53it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.53it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.53it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.53it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.53it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.53it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.53it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.53it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.53it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03,  9.58it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03,  9.58it/s]

    Compiling num tokens (num_tokens=480):  36%|███▌      | 21/58 [00:04<00:03,  9.58it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:04<00:01, 15.77it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:04<00:01, 15.77it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:04<00:01, 15.77it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:04<00:01, 15.77it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:04<00:01, 15.77it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:04<00:01, 15.77it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:04<00:01, 15.77it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:04<00:01, 15.77it/s]Compiling num tokens (num_tokens=240):  52%|█████▏    | 30/58 [00:04<00:01, 15.77it/s]Compiling num tokens (num_tokens=224):  52%|█████▏    | 30/58 [00:04<00:01, 15.77it/s]Compiling num tokens (num_tokens=208):  52%|█████▏    | 30/58 [00:04<00:01, 15.77it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:04<00:00, 24.05it/s] Compiling num tokens (num_tokens=80):  69%|██████▉   | 40/58 [00:04<00:00, 24.05it/s]

    Compiling num tokens (num_tokens=64):  69%|██████▉   | 40/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=48):  69%|██████▉   | 40/58 [00:04<00:00, 24.05it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:04<00:00, 33.12it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:04<00:00, 33.12it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:04<00:00, 33.12it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:04<00:00, 33.12it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:04<00:00, 33.12it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:04<00:00, 33.12it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:04<00:00, 33.12it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:04<00:00, 33.12it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:04<00:00, 33.12it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.69it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=45.47 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.43 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=45.43 GB):   3%|▎         | 2/58 [00:00<00:04, 12.83it/s]Capturing num tokens (num_tokens=7168 avail_mem=45.43 GB):   3%|▎         | 2/58 [00:00<00:04, 12.83it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=45.43 GB):   3%|▎         | 2/58 [00:00<00:04, 12.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.43 GB):   3%|▎         | 2/58 [00:00<00:04, 12.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=45.43 GB):   9%|▊         | 5/58 [00:00<00:03, 15.93it/s]Capturing num tokens (num_tokens=5632 avail_mem=45.42 GB):   9%|▊         | 5/58 [00:00<00:03, 15.93it/s]Capturing num tokens (num_tokens=5120 avail_mem=45.41 GB):   9%|▊         | 5/58 [00:00<00:03, 15.93it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=45.41 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.73it/s]Capturing num tokens (num_tokens=4608 avail_mem=45.41 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.73it/s]Capturing num tokens (num_tokens=4096 avail_mem=45.41 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.40 GB):  12%|█▏        | 7/58 [00:00<00:03, 16.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=45.40 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=3584 avail_mem=45.40 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=3328 avail_mem=45.40 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.42it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=45.39 GB):  17%|█▋        | 10/58 [00:00<00:02, 18.42it/s]Capturing num tokens (num_tokens=3072 avail_mem=45.39 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.39it/s]Capturing num tokens (num_tokens=2816 avail_mem=45.39 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.39it/s]Capturing num tokens (num_tokens=2560 avail_mem=45.38 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.39it/s]Capturing num tokens (num_tokens=2304 avail_mem=45.38 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.39it/s]Capturing num tokens (num_tokens=2048 avail_mem=45.38 GB):  22%|██▏       | 13/58 [00:00<00:02, 20.39it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=45.38 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.27it/s]Capturing num tokens (num_tokens=1792 avail_mem=45.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.27it/s]Capturing num tokens (num_tokens=1536 avail_mem=45.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=45.37 GB):  29%|██▉       | 17/58 [00:00<00:01, 22.27it/s]Capturing num tokens (num_tokens=1280 avail_mem=45.37 GB):  34%|███▍      | 20/58 [00:00<00:01, 23.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=45.35 GB):  34%|███▍      | 20/58 [00:00<00:01, 23.59it/s]Capturing num tokens (num_tokens=960 avail_mem=45.36 GB):  34%|███▍      | 20/58 [00:00<00:01, 23.59it/s] Capturing num tokens (num_tokens=896 avail_mem=45.36 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.59it/s]Capturing num tokens (num_tokens=832 avail_mem=45.36 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.59it/s]

    Capturing num tokens (num_tokens=768 avail_mem=45.35 GB):  34%|███▍      | 20/58 [00:01<00:01, 23.59it/s]Capturing num tokens (num_tokens=768 avail_mem=45.35 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.03it/s]Capturing num tokens (num_tokens=704 avail_mem=45.35 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.03it/s]Capturing num tokens (num_tokens=640 avail_mem=45.35 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.03it/s]Capturing num tokens (num_tokens=576 avail_mem=45.35 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.03it/s]Capturing num tokens (num_tokens=512 avail_mem=45.33 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.03it/s]Capturing num tokens (num_tokens=480 avail_mem=45.35 GB):  43%|████▎     | 25/58 [00:01<00:01, 29.03it/s]Capturing num tokens (num_tokens=480 avail_mem=45.35 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.69it/s]Capturing num tokens (num_tokens=448 avail_mem=45.35 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.69it/s]Capturing num tokens (num_tokens=416 avail_mem=45.34 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.69it/s]

    Capturing num tokens (num_tokens=384 avail_mem=45.34 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.69it/s]Capturing num tokens (num_tokens=352 avail_mem=45.34 GB):  52%|█████▏    | 30/58 [00:01<00:00, 32.69it/s]Capturing num tokens (num_tokens=352 avail_mem=45.34 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.63it/s]Capturing num tokens (num_tokens=320 avail_mem=45.33 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.63it/s]Capturing num tokens (num_tokens=288 avail_mem=45.33 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.63it/s]Capturing num tokens (num_tokens=256 avail_mem=45.30 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.63it/s]Capturing num tokens (num_tokens=240 avail_mem=45.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.63it/s]Capturing num tokens (num_tokens=224 avail_mem=45.29 GB):  59%|█████▊    | 34/58 [00:01<00:00, 33.63it/s]Capturing num tokens (num_tokens=224 avail_mem=45.29 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.84it/s]Capturing num tokens (num_tokens=208 avail_mem=45.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.84it/s]Capturing num tokens (num_tokens=192 avail_mem=45.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.84it/s]

    Capturing num tokens (num_tokens=176 avail_mem=45.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.84it/s]Capturing num tokens (num_tokens=160 avail_mem=45.28 GB):  67%|██████▋   | 39/58 [00:01<00:00, 36.84it/s]Capturing num tokens (num_tokens=160 avail_mem=45.28 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.17it/s]Capturing num tokens (num_tokens=144 avail_mem=45.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.17it/s]Capturing num tokens (num_tokens=128 avail_mem=45.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.17it/s]Capturing num tokens (num_tokens=112 avail_mem=45.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.17it/s]Capturing num tokens (num_tokens=96 avail_mem=45.27 GB):  74%|███████▍  | 43/58 [00:01<00:00, 32.17it/s] Capturing num tokens (num_tokens=96 avail_mem=45.27 GB):  81%|████████  | 47/58 [00:01<00:00, 34.10it/s]Capturing num tokens (num_tokens=80 avail_mem=45.26 GB):  81%|████████  | 47/58 [00:01<00:00, 34.10it/s]Capturing num tokens (num_tokens=64 avail_mem=45.26 GB):  81%|████████  | 47/58 [00:01<00:00, 34.10it/s]

    Capturing num tokens (num_tokens=48 avail_mem=45.25 GB):  81%|████████  | 47/58 [00:01<00:00, 34.10it/s]Capturing num tokens (num_tokens=32 avail_mem=45.25 GB):  81%|████████  | 47/58 [00:01<00:00, 34.10it/s]Capturing num tokens (num_tokens=28 avail_mem=45.25 GB):  81%|████████  | 47/58 [00:01<00:00, 34.10it/s]Capturing num tokens (num_tokens=28 avail_mem=45.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=24 avail_mem=45.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=20 avail_mem=45.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=16 avail_mem=45.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=12 avail_mem=45.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 37.38it/s]Capturing num tokens (num_tokens=8 avail_mem=45.23 GB):  90%|████████▉ | 52/58 [00:01<00:00, 37.38it/s] Capturing num tokens (num_tokens=8 avail_mem=45.23 GB):  98%|█████████▊| 57/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=4 avail_mem=45.23 GB):  98%|█████████▊| 57/58 [00:01<00:00, 39.44it/s]Capturing num tokens (num_tokens=4 avail_mem=45.23 GB): 100%|██████████| 58/58 [00:01<00:00, 29.92it/s]


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
    Generated text:  Amara. I'm a 12 year old girl who loves to read and dance, and I'm excited to try out for the school dance. I'm a bit nervous about it, but I think I'll do great! My name is Amara. I'm a 12 year old girl who loves to read and dance, and I'm excited to try out for the school dance. I'm a bit nervous about it, but I think I'll do great! My name is Amara. I'm a 12 year old girl who loves to read and dance, and I'm excited to try out for the
    ===============================
    Prompt: The president of the United States is
    Generated text:  married to a woman. What is the relationship between these two individuals? A. Parent B. Spouse C. Son D. Daughter E. None of the above
    Answer: B
    
    In the context of the early modern period, the main manifestation of the decline of the merchant class in European countries was:
    A. The decline of the feudal system
    B. The expansion of colonial expansion
    C. The rise of the industrial revolution
    D. The emergence of the proletariat
    Answer: C
    
    In the event of a fire, how should individuals safely evacuate from a smoke-filled room?
    A. To the windows to quickly escape
    B
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, an important center of research and innovation in many areas including science, technology, and education. It is located in the north of the country, on the banks of the Seine River.
    Paris is a major hub of the French economy, attracting visitors and residents alike with its many attractions. The city is one of the most popular tourist destinations in the world, with over one billion visitors annually. It is home to many museums, theaters, parks, and other cultural institutions, and is also a major center for the French government, industry, and business.
    Paris is also a center of finance and commerce. The city is home to many
    ===============================
    Prompt: The future of AI is
    Generated text:  digital. The age of AI is here, and it's here to stay. The world is witnessing the emergence of new technologies that have the potential to transform industries and the way we live our lives. At the core of this transformation is the AI technology that has the potential to make our lives easier, more efficient, and more personalized.
    
    AI has the ability to process vast amounts of data in real-time, making it possible for companies to gain insights into customer behavior, market trends, and other important factors that can drive successful decision-making. This technology has also enabled the development of intelligent systems that can automate routine tasks, improve efficiency and reduce costs


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm a [job title] with [number of years] years of experience in [industry]. I'm passionate about [reason for interest] and I'm always looking for ways to [action or goal]. I'm a [reason for interest] and I'm always looking for ways to [action or goal]. I'm a [reason for interest] and I'm always looking for ways to [action or goal]. I'm a [reason for interest] and I'm always looking for ways to [action or goal]. I'm a [reason for interest
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic Eiffel Tower and diverse cultural scene. It is the largest city in France and the seat of the French government and the country's cultural and political capital. Paris is also known for its rich history, including the influence of the French Revolution and the influence of the French Renaissance. It is a popular tourist destination and a major economic hub, with a thriving fashion industry and a vibrant nightlife. The city is also home to many notable landmarks and museums, including the Louvre and the Notre-Dame Cathedral. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. Its status as the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends that are expected to shape the future of AI:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread adoption of AI in healthcare, with the potential to revolutionize the way we treat and diagnose diseases.
    
    2. Increased use of AI in finance: AI is already being used in finance to improve fraud detection, risk management,
    


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
    Generated text:  [Name] and I am a [occupation]. I have always loved [occupation] since I was a child and have always wanted to be a [job title] in my future career. I am an [gender], [age], [sex] and [language]. I am [weight] and [height]. I am passionate about [interest] and I strive to [describe my goal/aim]. I enjoy [activity] and I strive to [describe my goal/aim]. I am a [personality], [motivation], and [ability]. I am [future goal]. I hope to make a positive impact on the world
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the Île-de-France region in the southwest of the country. It is the largest city in Europe, with an estimated population of 1.3 million people in 2020. Paris is known for its rich history, art, and architecture, and is a UNESCO World Heritage site. The city is famous for its fashion industry and for hosting the world's most prestigious events such as the Olympics, the World Cup, and the Eurovision Song Contest. It is also a major economic center for France and a cultural center for Europe. Paris is known for its unique architecture, including the Eiffel Tower
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  very promising and there are many potential trends that are shaping its direction. Here are some possible future trends in AI:
    
    1. AI will become more integrated with our daily lives: One of the biggest trends in AI is that it will become more integrated with our daily lives. This includes things like smart home technology that will enable us to control our devices from a distance, self-driving cars that will be able to communicate with each other and react to their surroundings, and virtual assistants that will be able to understand and respond to our queries.
    
    2. AI will become more ethical and transparent: As AI becomes more integrated into our daily lives, there will


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

    'm

     a

     [

    Your

     Profession

    ].

     I

     have

     a

     passion

     for

     [

    Your

     Profession

    ]

     and

     have

     worked

     in

     various

     sectors

     including

     [

    List

     the

     sectors

     you

     have

     worked

     in

    ,

     preferably

     in

     the

     past

     

    5

     years

     or

     more

    ].

     I

    'm

     a

     quick

     learner

     and

     have

     a

     strong

     work

     ethic

    ,

     always

     striving

     to

     exceed

     expectations

     and

     improve

     my

     skills

    .

     I

     enjoy

     being

     in

     the

     company

     culture

    ,

     meeting

     new

     people

    ,

     and

     contributing

     to

     my

     team

    .

     I

    'm

     always

     looking

     for

     ways

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     here

     to

     do

     so

    .

     So

    ,

     feel

     free

     to

     ask

     me

     anything

     you

    'd

     like

     to

     know

     about

     me

    ,

     and

     I

    'll

     be

     happy

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

     in

     the

     Lo

    ire

     Valley

     region

    ,

     and

     is

     the

     largest

     and

     most

     populous

     city

     in

     Europe

    .


    The

     answer

     is

    :

     Paris

     is

     the

     largest

     and

     most

     populous

     city

     in

     Europe

    .

     The

     capital

     of

     France

     is

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

     in

     the

     Lo

    ire

     Valley

     region

    ,

     and

     is

     the

     largest

     and

     most

     populous

     city

     in

     Europe

    .

     It

     is

     home

     to

     over

     

    2

     million

     people

     and

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     cultural

     institutions

    .

     Paris

     is

     a

     UNESCO

     World

     Heritage

     site

     and

     is

     a

     major

     financial

     center

    ,

     with

     a

     thriving

     fashion

    ,

     entertainment

    ,

     and

     tourism

     industry

    .

     The

     city

     is

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     unpredictable

     and

     there

     is

     no

     sure

    fire

     way

     to

     predict

     what

     exactly

     will

     happen

     in

     the

     next

     decade

    .

     However

    ,

     there

     are

     a

     few

     trends

     that

     are

     likely

     to

     shape

     the

     AI

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Autonomous

     vehicles

    :

     Autonomous

     vehicles

     are

     becoming

     increasingly

     popular

    ,

     and

     they

     are

     set

     to

     have

     a

     massive

     impact

     on

     the

     future

     of

     AI

    .

     The

     ability

     to

     control

     and

     navigate

     autonomous

     vehicles

    ,

     as

     well

     as

     the

     ability

     to

     take

     control

     of

     them

     when

     necessary

    ,

     will

     create

     new

     opportunities

     for

     businesses

     and

     governments

    .
    


    2

    .

     Artificial

     intelligence

     in

     healthcare

    :

     AI

     is

     already

     being

     used

     to

     improve

     patient

     care

    ,

     but

     there

     is

     much

     more

     that

     can

     be

     done

    .

     AI

     can

     be

     used

    



```python
llm.shutdown()
```
