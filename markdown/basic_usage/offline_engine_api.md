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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.34it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.33it/s]


    2026-05-09 11:23:26,902 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-09 11:23:26] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:40,  4.93s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:40,  4.93s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:05<00:40,  1.32it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:05<00:11,  3.96it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:05<00:11,  3.96it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:05<00:04,  8.91it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:05<00:04,  8.91it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:05<00:01, 14.45it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:05<00:01, 14.45it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:05<00:01, 14.45it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:05<00:01, 14.45it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:05<00:01, 14.45it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:05<00:01, 14.45it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:05<00:01, 14.45it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:05<00:01, 14.45it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:05<00:01, 14.45it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:05<00:01, 14.45it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:05<00:01, 14.45it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:05<00:00, 22.05it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:05<00:00, 22.05it/s]

    Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 22.05it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 30.58it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 30.58it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 30.58it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 30.58it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 30.58it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 30.58it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 30.58it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 30.58it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 10.30it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.79 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.26 GB):   3%|▎         | 2/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 15.57it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:03, 15.57it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 20.23it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.47it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.47it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.47it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.47it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.06 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.47it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.56it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.56it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.56it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.56it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.30it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.30it/s]

    Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.30it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.30it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 36.30it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 39.86it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 41.78it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 41.78it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 41.78it/s]

    Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 41.78it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 41.78it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 41.78it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=320 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=288 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 43.34it/s]Capturing num tokens (num_tokens=224 avail_mem=75.99 GB):  59%|█████▊    | 34/58 [00:01<00:00, 43.34it/s]Capturing num tokens (num_tokens=224 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.30it/s]

    Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=160 avail_mem=75.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  67%|██████▋   | 39/58 [00:01<00:00, 44.30it/s]Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=96 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.90it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=64 avail_mem=75.96 GB):  76%|███████▌  | 44/58 [00:01<00:00, 44.90it/s]Capturing num tokens (num_tokens=64 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.23it/s]

    Capturing num tokens (num_tokens=28 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  84%|████████▍ | 49/58 [00:01<00:00, 45.23it/s]Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.75it/s] Capturing num tokens (num_tokens=4 avail_mem=75.93 GB):  93%|█████████▎| 54/58 [00:01<00:00, 45.75it/s]Capturing num tokens (num_tokens=4 avail_mem=75.93 GB): 100%|██████████| 58/58 [00:01<00:00, 40.05it/s]


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
    Generated text:  Fred, and I've always been fascinated by the world of construction. It seems like a fascinating field that can take me anywhere, whether I'm building a bridge or creating a skyscraper. I've always been a student at a local college, but I've never taken any construction classes. So, I'm really interested in joining the construction industry. I'm eager to learn more about the industry, and I'm always open to hearing new ideas and techniques. I'm excited to meet new people and learn new things. Can you help me find out what construction companies are in my area and what kind of jobs are available there? And,
    ===============================
    Prompt: The president of the United States is
    Generated text:  seeking a reward from the media. He has 500 articles, each with an average of 15,000 word(s) of content. If he decides to reward 100 of these articles, what is the total number of word(s) of content that will be rewarded? To determine the total number of word(s) of content that will be rewarded, we need to follow these steps:
    
    1. Calculate the total number of word(s) of content in all the articles.
    2. Determine how many articles will be rewarded.
    3. Calculate the total number of word(s) of content that will be rewarded
    ===============================
    Prompt: The capital of France is
    Generated text: :
    A. Paris
    B. London
    C. Moscow
    D. New York
    Answer:
    
    A
    
    In the process of domestic sewage treatment, what are the main factors that determine the treatment efficacy of floating sludge?
    A. The toxicity of the pollutants
    B. The settling performance of the sludge
    C. The sedimentation performance of the sludge
    D. The sludge's microbial activity
    Answer:
    
    D
    
    Which of the following areas is known for its traditional furniture industry?
    A. Jingdezhen
    B. Suzhou
    C. Lin'an
    D. Suzhou, Jingdezhen, Lin'an
    
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. Artificial intelligence (AI) has become a key enabler of innovation and digital transformation. It is transforming industries, transforming our lives, and shaping the future of work. Machine learning (ML) and deep learning (DL) are enabling the next generation of AI and making it a practical reality. ML is enabling more complex AI that can understand, learn, and solve problems that had been previously unsolvable. DL has brought breakthroughs in natural language processing, computer vision, speech recognition, computerized decision making, etc. AI is increasingly being used in healthcare, finance, transportation, manufacturing, etc. There is a lot of research and


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


    Generated text:  [Name] and I'm a [occupation] who has been working in [industry] for [number] years. I'm passionate about [reason for passion], and I'm always looking for ways to [action or goal]. I'm a [character trait or quality] who is always [description of a trait or quality]. I'm [character description], and I'm excited to [reason for excitement]. I'm [character name]! [Name]! [Name]! [Name]! [Name]! [Name]! [Name]! [Name]! [Name]! [Name]! [Name]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris. 
    
    This statement is accurate and brief, providing the essential information about the capital city of France. It does not include any additional details or context beyond the core facts about Paris. If you need a more detailed or specific statement, please let me know, and I can expand it further. 
    
    For example, a more detailed statement could be: "Paris, the capital of France, is a bustling metropolis with a rich history, renowned for its art, culture, and cuisine." 
    
    If you have any other questions or need further clarification, feel free to ask! 
    
    Thank you. 
    
    ---
    
    **Note:** The statement "The
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation: As AI becomes more advanced, it is likely to become more and more integrated into our daily lives. This could lead to a greater reliance on automation in various industries, from manufacturing to transportation to healthcare.
    
    2. Enhanced privacy and security: As AI becomes more sophisticated, it is likely to require more data to function effectively. This could lead to increased concerns about privacy and security, as companies and governments must ensure
    


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
    Generated text:  [name], and I'm a [level] professional in the [field] industry. I bring over a decade of experience and a strong work ethic to this role. I'm an excellent communicator, able to deliver results that exceed expectations and always go the extra mile to ensure projects are completed to the highest standards. I am highly organized and detail-oriented, constantly striving to improve my skills and stay on top of industry trends. I am a team player, always willing to assist and support others. I'm passionate about my work and dedicated to providing exceptional customer service, and I am always looking for ways to expand my knowledge and experience. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known as the "City of Love." It is home to many notable landmarks, including the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral, and is a major tourist destination. The city is known for its art, culture, and cuisine, with Paris being one of the most important cultural centers in the world. Paris is also home to a diverse population of people of all backgrounds, and is the cultural and economic hub of France. It's a popular destination for international visitors, and its famous French cuisine, including croissants, fish and chips, and roulade, is highly regarded by the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be shaped by a number of factors, including advances in computing power, data availability, and the development of more advanced algorithms. Here are some potential trends that could shape the future of AI:
    
    1. Personalization: AI is likely to continue to become more personalized, with algorithms that can learn from user behavior and preferences to provide more accurate and relevant recommendations. This could lead to more efficient and effective use of resources, as well as better customer satisfaction and retention.
    
    2. Autonomous vehicles: Autonomous vehicles may become more prevalent as AI technology improves and becomes more widespread. This could lead to safer and more efficient transportation, as well as reduced


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

    'm

     an

     [

    occupation

    ]

     who

     specializes

     in

     [

    job

     title

     or

     role

    ].

     I

    've

     always

     had

     a

     natural

     talent

     for

     [

    specific

     skill

     or

     ability

    ],

     and

     over

     the

     years

    ,

     I

    've

     hon

    ed

     it

     into

     [

    mention

     any

     particular

     skill

     or

     expertise

    ].

     I

    've

     worked

     with

     a

     variety

     of

     clients

     and

     have

     a

     strong

     sense

     of

     [

    ability

     or

     interest

    ],

     which

     has

     allowed

     me

     to

     excel

     in

     my

     field

    .

     I

    'm

     always

     looking

     to

     learn

     and

     grow

    ,

     and

     I

    'm

     excited

     to

     see

     where

     my

     skills

     can

     take

     me

     in

     the

     future

    .

     What

    's

     something

     I

     can

     tell

     my

     potential

     clients

     or

     colleagues

     about

     my

     background

     and

     what

     sets

     me

     apart

    ?

     [

    Please

     provide

    
    
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

     city

     of

     France

    ,

     located

     on

     the

     Mediterranean

     coast

     near

     the

     mouth

     of

     the

     River

     Se

    ine

    .

     The

     city

     is

     known

     for

     its

     historical

     landmarks

    ,

     vibrant

     culture

    ,

     and

     fashion

     scene

    .

     It

    's

     a

     bustling

     met

    ropolis

     with

     a

     rich

     culture

     and

     a

     large

     population

     of

     approximately

     

    2

    .

    3

     million

     people

    .

     Paris

     is

     also

     the

     seat

     of

     the

     French

     government

     and

     home

     to

     numerous

     museums

    ,

     art

     galleries

    ,

     and

     theaters

    .

     The

     city

     is

     known

     for

     its

     iconic

     architecture

    ,

     including

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

     Paris

     is

     a

     city

     of

     contrasts

    ,

     with

     its

     modern

     skys

    crap

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

     and

     potentially

     revolutionary

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

     AI

     efficiency

     and

     accuracy

    :

     As

     AI

     technologies

     continue

     to

     advance

    ,

     the

     efficiency

     and

     accuracy

     of

     AI

     solutions

     will

     continue

     to

     improve

    .

     This

     means

     that

     AI

    -powered

     solutions

     will

     become

     increasingly

     reliable

     and

     consistent

    ,

     making

     them

     a

     more

     reliable

     choice

     for

     a

     wide

     range

     of

     applications

    .
    


    2

    .

     AI

     democrat

    ization

    :

     AI

     will

     continue

     to

     become

     more

     accessible

     to

     a

     wider

     range

     of

     people

    .

     This

     means

     that

     AI

     solutions

     will

     become

     more

     accessible

     to

     those

     who

     were

     previously

     excluded

     from

     the

     technology

    ,

     such

     as

     people

     with

     disabilities

    ,

     those

     who

     live

     in

     remote

     areas

    ,

     or

     those

     who

     are

     new

     to

     AI

    .
    


    



```python
llm.shutdown()
```
