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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.56it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:06,  4.32s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.50it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.49it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.49it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.05it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.17it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 24.42it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 24.42it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 24.42it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 24.42it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 24.42it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 24.42it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 24.42it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 24.42it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 24.42it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 24.42it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 24.42it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 33.59it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 33.59it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 33.59it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 33.59it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 33.59it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 33.59it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 33.59it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 33.59it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 11.61it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=71.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.35it/s]Capturing num tokens (num_tokens=7168 avail_mem=71.78 GB):   3%|▎         | 2/58 [00:00<00:02, 19.35it/s]Capturing num tokens (num_tokens=6656 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.35it/s]Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   3%|▎         | 2/58 [00:00<00:02, 19.35it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=5632 avail_mem=71.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=5120 avail_mem=71.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=4608 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):   9%|▊         | 5/58 [00:00<00:02, 22.61it/s]Capturing num tokens (num_tokens=4096 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.36it/s]Capturing num tokens (num_tokens=3840 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.36it/s]Capturing num tokens (num_tokens=3584 avail_mem=71.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.36it/s]Capturing num tokens (num_tokens=3328 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.36it/s]Capturing num tokens (num_tokens=3072 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.36it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.36it/s]Capturing num tokens (num_tokens=2816 avail_mem=71.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.13it/s]Capturing num tokens (num_tokens=2560 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.13it/s]Capturing num tokens (num_tokens=2304 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.13it/s]Capturing num tokens (num_tokens=2048 avail_mem=71.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.13it/s]Capturing num tokens (num_tokens=1792 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.13it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.13it/s]Capturing num tokens (num_tokens=1536 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=71.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=71.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.93it/s]Capturing num tokens (num_tokens=960 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.93it/s] Capturing num tokens (num_tokens=896 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.93it/s]

    Capturing num tokens (num_tokens=832 avail_mem=71.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.93it/s]Capturing num tokens (num_tokens=832 avail_mem=71.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=768 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=704 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=640 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=576 avail_mem=71.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.34it/s]Capturing num tokens (num_tokens=512 avail_mem=71.68 GB):  50%|█████     | 29/58 [00:00<00:00, 43.63it/s]Capturing num tokens (num_tokens=480 avail_mem=71.70 GB):  50%|█████     | 29/58 [00:00<00:00, 43.63it/s]Capturing num tokens (num_tokens=448 avail_mem=71.70 GB):  50%|█████     | 29/58 [00:00<00:00, 43.63it/s]Capturing num tokens (num_tokens=416 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 43.63it/s]Capturing num tokens (num_tokens=384 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 43.63it/s]

    Capturing num tokens (num_tokens=352 avail_mem=71.69 GB):  50%|█████     | 29/58 [00:00<00:00, 43.63it/s]Capturing num tokens (num_tokens=352 avail_mem=71.69 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.32it/s]Capturing num tokens (num_tokens=320 avail_mem=71.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.32it/s]Capturing num tokens (num_tokens=288 avail_mem=71.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.32it/s]Capturing num tokens (num_tokens=256 avail_mem=71.68 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.32it/s]Capturing num tokens (num_tokens=240 avail_mem=71.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.32it/s]Capturing num tokens (num_tokens=224 avail_mem=71.67 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.32it/s]Capturing num tokens (num_tokens=224 avail_mem=71.67 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.56it/s]Capturing num tokens (num_tokens=208 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:00<00:00, 46.56it/s]Capturing num tokens (num_tokens=192 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=176 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=160 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.56it/s]

    Capturing num tokens (num_tokens=144 avail_mem=71.66 GB):  67%|██████▋   | 39/58 [00:01<00:00, 46.56it/s]Capturing num tokens (num_tokens=144 avail_mem=71.66 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=128 avail_mem=71.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=112 avail_mem=71.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=96 avail_mem=71.65 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.29it/s] Capturing num tokens (num_tokens=80 avail_mem=71.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  76%|███████▌  | 44/58 [00:01<00:00, 47.29it/s]Capturing num tokens (num_tokens=64 avail_mem=71.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.17it/s]Capturing num tokens (num_tokens=48 avail_mem=71.64 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.17it/s]Capturing num tokens (num_tokens=32 avail_mem=71.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.17it/s]Capturing num tokens (num_tokens=28 avail_mem=71.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.17it/s]Capturing num tokens (num_tokens=24 avail_mem=71.63 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.17it/s]

    Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  84%|████████▍ | 49/58 [00:01<00:00, 47.17it/s]Capturing num tokens (num_tokens=20 avail_mem=71.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.44it/s]Capturing num tokens (num_tokens=16 avail_mem=71.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.44it/s]Capturing num tokens (num_tokens=12 avail_mem=71.62 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.44it/s]Capturing num tokens (num_tokens=8 avail_mem=71.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.44it/s] Capturing num tokens (num_tokens=4 avail_mem=71.61 GB):  93%|█████████▎| 54/58 [00:01<00:00, 47.44it/s]Capturing num tokens (num_tokens=4 avail_mem=71.61 GB): 100%|██████████| 58/58 [00:01<00:00, 42.13it/s]


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
    Generated text:  Zoubad, a neurologist. I treat my patients with a variety of conditions, including epilepsy, anxiety, depression, sleep disorders, and more.
    I'm an associate professor at the University of Colorado School of Medicine, where I'm a licensed physician. I studied psychology at the University of California, Berkeley.
    I'm a patient advocate and a frequent speaker on mental health issues. I'm a member of the American Psychiatric Association and a member of the American Association of Sleep Medicine.
    My background and clinical expertise span a variety of neurological and psychiatric conditions. I have extensive experience in the use of cognitive behavioral therapy and other evidence-based
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many armed guards to have at his presidential mansion. The president believes that there are x people in the United States and that there are about 100 guards for every x people. The president wants this ratio to be as close to 1 as possible. How many guards should the president have? To determine the number of guards the president should have, we start by understanding the given ratio and the relationship between the number of people and guards. The ratio of people to guards is given by the fraction:
    
    \[
    \frac{x}{100}
    \]
    
    The president wants this ratio to be as close to 
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris, which is a long city that is surrounded by a wall. During the French Revolution, the citizens were forced to leave Paris in order to protect their lives and property from the invading armies. The Wall was built by the government to protect the citizens. The Wall was destroyed by the revolutionaries. The city has many historical buildings and museums. The Wall is the only structure that still stands today. People who visited Paris during the French Revolution were very impressed by the beautiful architecture that was destroyed by the revolutionaries. What happened to the Wall? The Wall was destroyed by the revolutionaries. The Revolutionaries were led by Napoleon Bonaparte
    ===============================
    Prompt: The future of AI is
    Generated text:  about the use of the human mind, not computers
    
    Pete Dyer, CEO of Xai.ai
    
    If you’re reading this, you’ve probably already seen a movie about a robot who tries to understand human language. Or a story about a person with a kind of autism, who seems to have a brain that’s been transplant from a dog.
    
    These scenarios might have been easier to imagine in the past, when computers were still not powerful enough to produce the kind of sophisticated understanding that an AI could produce today. But that’s changing as the field of artificial intelligence has become much more capable.
    
    Today, you can use a computer to


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


    Generated text:  [Name] and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [job title] at [company name], and I'm passionate about [job title] and [job title] at [company name]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [favorite hobby or activity], and I'm always looking for new ways to explore and discover new things. What's your favorite book or movie? I love [favorite book or movie],
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. The city is also famous for its cuisine, fashion, and art scene. Paris is a major tourist destination and a cultural hub in Europe. It is home to many world-renowned museums, theaters, and art galleries. The city is also known for its fashion industry, with many famous designers and boutiques. Paris is a city of contrasts, with its modern
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely trends that could be expected in the future:
    
    1. Increased automation: As AI becomes more advanced, it is likely to become more and more integrated into our daily lives. This could lead to a significant increase in automation, where machines will take on many of the tasks that humans currently do. This could lead to a reduction in the need for human workers, which could have a positive impact on the economy.
    
    2. AI ethics and privacy: As AI becomes more advanced, there
    


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
    Generated text:  [Name] and I'm a [Grade level] student at [School name]. I'm [Occupation] and [Name] has been my best friend since [Year]. I enjoy [My favorite hobby or interest]. I love [A sport or hobby you've been playing for a long time]. I'm [Your age]. I'm a [Your favorite color]. [Your favorite quote or saying]. My best friend is [Friend's name] and she's always there for me. I'm [Your grade level] and I enjoy playing [My favorite game]. What about you? How's your day going? What's your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is known for its iconic Eiffel Tower and fashion-conscious fashionistas. It is also home to the world-renowned Louvre Museum, and the iconic Eiffel Tower. Paris is a vibrant city with a diverse population, and it is an important cultural and economic center in Europe. Its skyline is filled with tall buildings and monuments, and it is home to a wide range of art, culture, and cuisine. Despite its beautiful and historic landmarks, Paris is also a bustling city with a high turnover of tourists. It is a vibrant and diverse city with a strong sense of community. The city is also known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to involve several trends that are currently being explored and developed. Here are some of the most significant trends that may shape the AI landscape in the coming years:
    
    1. Increased AI Transparency: As AI continues to become more powerful, there will be an increasing demand for transparency in its development, deployment, and operation. This may lead to the creation of more accessible and understandable AI systems, as well as more transparent regulatory frameworks for AI development and deployment.
    
    2. Increased AI Ethics: With the advent of AI-powered technologies, there will be an increased demand for ethical considerations to be incorporated into the design and development of these technologies. This may lead


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

     [

    Type

    ]

     who

     has

     [

    Number

    ]

     years

     of

     experience

     in

     [

    Industry

    /

    Field

    ].

     I

     specialize

     in

     [

    Most

     Common

     Occupation

    ].

     I

     have

     a

     unique

     skill

    set

     that

     makes

     me

     stand

     out

     among

     my

     peers

     and

     I

     am

     always

     looking

     to

     improve

     my

     knowledge

     and

     skills

     to

     be

     the

     best

     in

     my

     field

    .

     I

     am

     always

     eager

     to

     learn

     and

     grow

    ,

     and

     I

     am

     committed

     to

     being

     a

     valuable

     member

     of

     the

     team

    .

     What

     is

     your

     name

     and

     what

     type

     of

     work

     do

     you

     do

    ?

     Are

     you

     interested

     in

     learning

     more

     about

     me

    ?

     If

     so

    ,

     what

     would

     you

     like

     to

     know

     about

     me

    ?

     I

     would

     like

     to

     make

     a

     good

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

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

     vibrant

     culture

    .

     It

     is

     the

     oldest

     capital

     city

     in

     the

     world

     and

     is

     home

     to

     many

     iconic

     landmarks

     such

     as

     the

     E

    iff

    el

     Tower

     and

     Notre

     Dame

     Cathedral

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

     and

     has

     played

     a

     significant

     role

     in

     the

     development

     of

     Western

     civilization

    .

     It

     is

     a

     world

    -ren

    owned

     tourist

     destination

     and

     a

     popular

     destination

     for

     business

     and

     leisure

     activities

    .

     Paris

     is

     home

     to

     numerous

     museums

    ,

     theaters

    ,

     and

     art

     galleries

    ,

     making

     it

     a

     popular

     destination

     for

     art

     lovers

     and

     visitors

    .

     The

     city

     is

     also

     known

     for

     its

     wine

     industry

     and

     has

     produced

     some

     of

     the

     world

    ’s

     finest

     wines

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     rapidly

     evolving

     and

     will

     continue

     to

     drive

     significant

     changes

     in

     various

     sectors

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     are

     expected

     to

     shape

     the

     landscape

     in

     the

     coming

     years

    :
    


    1

    .

     Adv

    ancements

     in

     machine

     learning

    :

     As

     AI

     systems

     become

     more

     sophisticated

    ,

     they

     will

     be

     able

     to

     learn

     and

     adapt

     to

     new

     situations

     more

     effectively

     than

     ever

     before

    .

     This

     will

     enable

     them

     to

     perform

     tasks

     that

     are

     currently

     beyond

     their

     capabilities

    ,

     such

     as

     speech

     recognition

     and

     image

     recognition

    .
    


    2

    .

     Increased

     integration

     of

     AI

     with

     other

     technologies

    :

     As

     AI

     becomes

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     robotics

    ,

     autonomous

     vehicles

    ,

     and

     quantum

     computing

    ,

     the

     possibilities

     for

     new

     applications

     will

     continue

     to

     expand

    .

     For

     example

    



```python
llm.shutdown()
```
