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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.86it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.86it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:09,  4.38s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.41it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.41it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]

    Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.87it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 15.85it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]

    Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 23.95it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:05<00:00, 23.95it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:05<00:00, 23.95it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:05<00:00, 32.93it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:05<00:00, 32.93it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:05<00:00, 32.93it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:05<00:00, 32.93it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:05<00:00, 32.93it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:05<00:00, 32.93it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:05<00:00, 32.93it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:05<00:00, 32.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:03, 15.53it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 15.53it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 15.53it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:03, 15.53it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 20.28it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.92it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.92it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.92it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.92it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.92it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 25.92it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.85it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.85it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.85it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.85it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 31.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]

    Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s] Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=832 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.32it/s]Capturing num tokens (num_tokens=832 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=640 avail_mem=71.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=576 avail_mem=71.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=480 avail_mem=71.69 GB):  41%|████▏     | 24/58 [00:00<00:00, 40.84it/s]Capturing num tokens (num_tokens=480 avail_mem=71.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.12it/s]Capturing num tokens (num_tokens=448 avail_mem=71.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.12it/s]

    Capturing num tokens (num_tokens=416 avail_mem=71.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.12it/s]Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.12it/s]Capturing num tokens (num_tokens=352 avail_mem=71.68 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.12it/s]Capturing num tokens (num_tokens=320 avail_mem=71.68 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.12it/s]Capturing num tokens (num_tokens=288 avail_mem=71.67 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.12it/s]Capturing num tokens (num_tokens=288 avail_mem=71.67 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=256 avail_mem=71.67 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=224 avail_mem=71.66 GB):  62%|██████▏   | 36/58 [00:00<00:00, 46.09it/s]Capturing num tokens (num_tokens=208 avail_mem=71.66 GB):  62%|██████▏   | 36/58 [00:01<00:00, 46.09it/s]Capturing num tokens (num_tokens=192 avail_mem=71.66 GB):  62%|██████▏   | 36/58 [00:01<00:00, 46.09it/s]

    Capturing num tokens (num_tokens=192 avail_mem=71.66 GB):  71%|███████   | 41/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=176 avail_mem=71.66 GB):  71%|███████   | 41/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  71%|███████   | 41/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=144 avail_mem=71.65 GB):  71%|███████   | 41/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=128 avail_mem=71.65 GB):  71%|███████   | 41/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  71%|███████   | 41/58 [00:01<00:00, 46.71it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=96 avail_mem=71.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.15it/s] Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.15it/s]

    Capturing num tokens (num_tokens=48 avail_mem=71.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 42.15it/s]Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=28 avail_mem=71.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=24 avail_mem=71.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=12 avail_mem=71.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.57it/s]Capturing num tokens (num_tokens=12 avail_mem=71.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.14it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.14it/s] Capturing num tokens (num_tokens=4 avail_mem=71.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.14it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 40.28it/s]


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
    Generated text:  Valeriu, and I am a photographer, artist, and writer based in San Francisco, California. I have been practicing photography since 2010, and have also exhibited my work across Europe, Australia, and the United States, including galleries in New York, Paris, and Milan.
    My work is rooted in an exploration of everyday moments, the beauty and chaos of human experience, and the social and cultural legacies of mass production. My photography often depicts the mundane and the mundane in fascinating ways, showing how life can be both intense and fluid.
    My style, which blends elements of street photography, documentary, and fine art
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the Senate. The president of the United States is also a member of the House of Representatives. Which statement is true about the House of Representatives and the Senate? A  The President is a member of the Senate, but not the House of Representatives. B  The President is a member of both the House of Representatives and the Senate. C  The President is a member of neither the House of Representatives nor the Senate. D  The President is a member of both the House of Representatives and the Senate. A, B, C, and D are all possible answers. To determine the correct statement, we need to analyze the
    ===============================
    Prompt: The capital of France is
    Generated text: :
    
    A. Paris  
    B. London  
    C. Milan  
    D. Rome  
    E. Edinburgh  
    
    My answer is A. Paris. Is my answer correct?
    To determine the capital of France, let's review the options provided:
    
    A. Paris  
    B. London  
    C. Milan  
    D. Rome  
    E. Edinburgh
    
    The capital of France is Paris. Therefore, the correct answer is:
    
    A. Paris
    
    To verify, let's consider the cities listed:
    - London is the capital of the United Kingdom.
    - Milan is the capital of Italy.
    - Rome is the capital of Italy.
    - Edinburgh is the capital of Scotland
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. But it will come at a cost. The environmental impact of AI and the automation of automation can be so great it has to be accounted for.
    According to the International Energy Agency, the cost of CO2 emissions from electric power generation will increase from 490 billion tonnes in 2015 to 680 billion in 2040. That would represent a 130% increase in emissions.
    In the United States, the costs of CO2 emissions from electric power generation will increase from 140 billion tonnes in 2015 to 180 billion in 2


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


    Generated text:  [Name] and I am a [Age] year old [Gender] [Occupation]. I am a [Skill] [Ability] who has been [Career] for [Number of Years] years. I am passionate about [What I Love to Do] and I am always looking for ways to [What I Want to Improve]. I am a [Personality] person who is [What I Like/Dislike About Myself]. I am [What I Hope to Achieve in the Future]. I am excited to meet you and learn more about you. How can I help you today? [Name] [Age] [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic hub, known for its rich history, art, and cuisine. It is the largest city in France by population and is a major tourist destination. The city is also home to many international organizations and institutions, including the European Parliament and the United Nations. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into the urban landscape. Its status as the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction. Here are some of the most likely trends:
    
    1. Increased use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes, reduce costs, and increase efficiency. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare.
    
    2. AI in manufacturing: AI is already being used in manufacturing to improve efficiency, reduce costs, and increase productivity. As AI technology continues to improve, we can expect to see even more widespread use of AI in manufacturing.
    
    3. AI in finance:
    


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
    Generated text:  [Your Name] and I'm a young entrepreneur with a passion for [Your passion]. I'm currently pursuing my Master's degree in [Your field of study], with a strong background in [relevant skills]. What's your story? I've always been driven to create something unique and innovative, driven by a desire to make a positive impact in the world. I'm ready to bring my ideas to life and turn them into reality, one step at a time. I'm excited to meet you! [Your Name] (Leave the name blank) [Your Contact Information] [Your Social Media Links] [Your Skills and Expertise]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, located in the central region of the country. It is the largest city in the country and has a rich history and culture. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. The city also has a diverse population of over 2 million inhabitants, and is an important economic and cultural center in France. It is a popular tourist destination, hosting international events, festivals, and cultural activities. Paris is considered one of the most beautiful cities in the world and has been a UNESCO World Heritage Site since 1985. The city is famous for its fashion
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  poised to be a significant part of the tech industry. Here are some potential trends in AI that could play a major role:
    
    1. Autonomous vehicles: With the increasing number of autonomous vehicles on the road, we can expect to see more and more self-driving cars in the future. AI-powered systems will need to learn how to recognize and avoid obstacles, handle complex traffic conditions, and communicate with other vehicles in real-time. As these systems become more advanced, they could even be autonomous in their decision-making processes.
    
    2. Deep learning and artificial intelligence: With the development of deep learning algorithms, AI is getting even more sophisticated. Deep learning will


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

     skilled

    [

    insert

     skill

     or

     talent

    ]

     who

     loves

     to

     [

    insert

     hobby

     or

     passion

    ].

     I

    'm

     passionate

     about

     [

    insert

     something

     that

     reflects

     your

     interests

     and

     passions

    ,

     like

     your

     home

     life

    ,

     hobbies

    ,

     or

     achievements

    ].


    As

     a

     [

    insert

     your

     profession

    ],

     [

    Your

     Name

    ]

     has

     been

     [

    insert

     number

     of

     years

     of

     experience

    ]

     years

     in

     this

     field

    .

     I

    'm

     known

     for

     [

    insert

     one

     or

     two

     words

     that

     reflect

     your

     unique

     qualities

    ,

     like

     [

    insert

     a

     trait

     of

     yours

    ]]

     and

     have

     a

     passion

     for

     [

    insert

     something

     that

     reflects

     your

     interests

     and

     passions

    ,

     like

     your

     home

     life

    ,

     hobbies

    ,

     or

     achievements

    ].

     I

    'm

     always

     striving

     to

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     in

     the

     north

    western

     region

     of

     the

     country

    ,

     and

     is

     known

     as

     the

     "

    City

     of

     Love

    "

     due

     to

     its

     rich

     historical

     and

     cultural

     heritage

    .

     Paris

     is

     also

     known

     as

     the

     "

    City

     of

     Light

    "

     due

     to

     its

     vibrant

     and

     colorful

     neighborhoods

     and

     its

     influence

     on

     popular

     culture

    .

     Its

     ancient

     architecture

    ,

     stunning

     museums

    ,

     and

     unique

     fashion

     scene

     make

     it

     a

     popular

     destination

     for

     tourists

     and

     locals

     alike

    .

     Paris

     has

     a

     rich

     history

     dating

     back

     to

     the

     Roman

     Empire

     and

     has

     been

     a

     major

     center

     of

     French

     culture

     and

     politics

     for

     over

     

    1

    ,

    5

    0

    0

     years

    .

     Today

    ,

     it

     remains

     a

     major

     center

     of

     business

    ,

     entertainment

    ,

     and

     tourism

    ,

     with

     many

     famous

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     bright

     and

     is

     likely

     to

     continue

     to

     evolve

     in

     the

     following

     ways

    :

     

    1

    )

     AI

     will

     continue

     to

     improve

     in

     accuracy

     and

     precision

    .

     

    2

    )

     AI

     will

     become

     more

     ethical

     and

     responsible

     as

     more

     people

     and

     governments

     demand

     it

    .

     

    3

    )

     AI

     will

     be

     more

     integrated

     into

     everyday

     life

    ,

     from

     home

     automation

     to

     healthcare

    .

     

    4

    )

     AI

     will

     continue

     to

     play

     a

     critical

     role

     in

     shaping

     the

     future

     of

     work

     and

     education

    .

     

    5

    )

     AI

     will

     continue

     to

     be

     used

     for

     good

    ,

     with

     researchers

     and

     developers

     working

     to

     advance

     and

     improve

     its

     capabilities

     in

     areas

     like

     climate

     change

     and

     mental

     health

    .

     

    6

    )

     AI

     will

     continue

     to

     be

     an

     important

     tool

     for

     solving

     global

    



```python
llm.shutdown()
```
