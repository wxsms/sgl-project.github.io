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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.57it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  2.57it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:50,  4.04s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:33,  1.60it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.76it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.76it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.62it/s]

    Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.62it/s]Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 16.98it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 16.98it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 16.98it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 16.98it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 16.98it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 16.98it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 16.98it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 16.98it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 16.98it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 16.98it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 16.98it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 25.52it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 25.52it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 25.52it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 25.52it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 25.52it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 25.52it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 25.52it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 25.52it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 25.52it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 25.52it/s]

    Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 25.52it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 34.68it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 34.68it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 34.68it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 34.68it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 34.68it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 34.68it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 34.68it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 34.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.25it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.30 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.27 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.27 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.26 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   3%|▎         | 2/58 [00:00<00:02, 18.94it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.10 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.09 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):   9%|▊         | 5/58 [00:00<00:02, 22.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.04it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.07 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.06 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.05 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.03 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s]Capturing num tokens (num_tokens=960 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s] Capturing num tokens (num_tokens=896 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.36it/s]Capturing num tokens (num_tokens=832 avail_mem=76.04 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=768 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=704 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=640 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=576 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.85it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=480 avail_mem=76.03 GB):  50%|█████     | 29/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=416 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=384 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:00<00:00, 44.23it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:00<00:00, 44.23it/s]Capturing num tokens (num_tokens=352 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=320 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=288 avail_mem=76.01 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=256 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=240 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=224 avail_mem=76.00 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  59%|█████▊    | 34/58 [00:00<00:00, 45.94it/s]Capturing num tokens (num_tokens=208 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.47it/s]Capturing num tokens (num_tokens=192 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:00<00:00, 47.47it/s]Capturing num tokens (num_tokens=176 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=160 avail_mem=75.99 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.47it/s]

    Capturing num tokens (num_tokens=144 avail_mem=75.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  69%|██████▉   | 40/58 [00:01<00:00, 47.47it/s]Capturing num tokens (num_tokens=128 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.97it/s]Capturing num tokens (num_tokens=112 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.97it/s]Capturing num tokens (num_tokens=96 avail_mem=75.98 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.97it/s] Capturing num tokens (num_tokens=80 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.97it/s]Capturing num tokens (num_tokens=64 avail_mem=75.97 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.97it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  78%|███████▊  | 45/58 [00:01<00:00, 47.97it/s]Capturing num tokens (num_tokens=48 avail_mem=75.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=32 avail_mem=75.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=28 avail_mem=75.96 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=24 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]

    Capturing num tokens (num_tokens=20 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  86%|████████▌ | 50/58 [00:01<00:00, 47.71it/s]Capturing num tokens (num_tokens=16 avail_mem=75.95 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=12 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=8 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.92it/s] Capturing num tokens (num_tokens=4 avail_mem=75.94 GB):  95%|█████████▍| 55/58 [00:01<00:00, 47.92it/s]Capturing num tokens (num_tokens=4 avail_mem=75.94 GB): 100%|██████████| 58/58 [00:01<00:00, 42.43it/s]


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
    Generated text:  Janie and I'm a teenager from Atlanta. I have brown hair and blue eyes. I'm popular among my peers and I like to go to movies and games. I'm really interested in dinosaurs. What do you think dinosaurs are?
    
    Dinosaurs are gigantic reptiles that lived millions of years ago. They lived in the Mesozoic era, which lasted from about 250 million years to 65 million years ago. They were the prehistoric humans we call dinosaurs. The name "dinosaur" is a very old word that comes from the Greek word "kata" meaning "large" and "dinos
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide how many military jets he should buy. He found out from his experts that the cost of a jet increases by 15% each year. If the cost of a jet today is $120,000, how much will the jets cost in 3 years? Express your answer to the nearest hundred.
    To determine the cost of a jet in 3 years, we need to calculate the cost step by step, taking into account the annual 15% increase.
    
    1. **Initial Cost:**
       The cost of the jet today is $120,000.
    
    2. **
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It was built between 1792 and 1804 by the Paris Commune. The construction was slow due to the economic crisis and there were also unrest due to the communal living methods in the countryside. Even though the Commune was a very successful social movement, it led to a strong desire for independence for the Parisian people. To achieve that goal, the Commune and its allies launched a campaign of self-defence. The first attack on the enemy was a fire at the Palais de Justice. The following year, the Commune attacked the Bastille. The Bastille was destroyed by the Continental
    ===============================
    Prompt: The future of AI is
    Generated text:  uncertain
    
    The future of AI is uncertain
    
    There are two types of AI that have the potential to have a profound impact on humanity. The first type of AI is responsible for creating and improving things that we can see, such as self-driving cars. The second type of AI, which is not as visible, is responsible for creating and improving things that we can’t see, such as prosthetics, games, and voice recognition. While these AI systems are capable of automating human tasks, they also have the potential to create new forms of human-technology hybrids, such as artificial intelligence and consciousness.
    
    Artificial intelligence is defined by the ability


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


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich history dating back to the Roman Empire and the Middle Ages. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its fashion industry, with Paris Fashion Week being one of the largest in the world. Paris is a vibrant and cosmopolitan city with a diverse population and a rich cultural heritage. It is a popular tourist destination and a major economic center in Europe. The city is also home to many famous museums, including
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of this technology in the coming years. Here are some of the most likely trends that could shape the future of AI:
    
    1. Increased Use of AI in Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to improve, we can expect to see even more widespread use of AI in healthcare in the coming years.
    
    2. Increased Use of AI in Finance: AI is already being used in finance to improve fraud detection and risk management. As AI technology continues to improve, we can expect to see even more widespread
    


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
    Generated text:  [Name] and I'm a self-employed [field] who enjoys [job role] and [other role if applicable]. What brings you to this position, and what are some of your most memorable experiences in this role? 
    
    I have [number of years in this position] years of experience in [specific field or role] and have always been passionate about [something related to the job]. What are your career aspirations and how do you see yourself progressing within this role? 
    
    What brings you to this position, and what are some of your most memorable experiences in this role? 
    
    I have [number of years in this position] years
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, also known as "The City of Light". It is a bustling metropolis with a rich history and a beautiful architecture. The city is home to iconic landmarks such as the Eiffel Tower and the Louvre Museum, as well as a diverse array of cultural attractions. Paris is a popular tourist destination with millions of visitors each year. The city is known for its fashion, food, and art scenes, and is a popular destination for business, leisure, and entertainment. Despite its size, Paris is an incredibly vibrant and dynamic city, with a strong sense of community and a rich cultural identity. The city is a symbol of France
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by a wide range of potential trends that could shape how AI is used in various industries, from healthcare and finance to transportation and entertainment. Some of the key trends that are likely to be seen in the future include:
    
    1. Autonomous vehicles: As autonomous driving technology becomes more advanced and widespread, we can expect to see more and more cars on the road being designed to be self-driving. This could lead to a reduction in the need for human drivers, and ultimately, an increase in the efficiency and safety of transportation systems.
    
    2. Artificial intelligence in healthcare: AI is already being used in medicine to analyze medical images, diagnose


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

    insert

     your

     occupation

    ]

     who

     has

     a

     passion

     for

     [

    insert

     something

     you

    're

     passionate

     about

    ,

     such

     as

     photography

     or

     writing

    ].

     I

    've

     always

     been

     fascinated

     by

     the

     stories

     of

     people

     who

     have

     overcome

     great

     challenges

    ,

     and

     I

    'm

     here

     to

     share

     those

     stories

     with

     you

    .

     What

    's

     your

     favorite

     book

     or

     movie

    ,

     and

     why

     do

     you

     think

     it

     reson

    ates

     with

     you

    ?


    [

    Your

     Name

    ]

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Love

    .

     The

     city

     is

     home

     to

     many

     famous

     landmarks

     such

     as

     Notre

    -D

    ame

     Cathedral

    ,

     the

     Arc

     de

     Tri

    omp

    he

    ,

     the

     Lou

    vre

     Museum

    ,

     the

     E

    iff

    el

     Tower

    ,

     and

     the

     Paris

     Opera

    .

     Paris

     is

     also

     known

     as

     "

    La

     Vie

     en

     Rose

    "

     due

     to

     its

     vibrant

     and

     colorful

     streets

    ,

     and

     is

     a

     popular

     destination

     for

     tourists

     and

     artists

     alike

    .

     The

     city

     is

     known

     for

     its

     rich

     history

     and

     culture

    ,

     and

     has

     played

     a

     significant

     role

     in

     French

     politics

     and

     culture

     for

     centuries

    .

     Its

     annual

     Le

     Se

    jour

    ne

    ,

     a

     famous

     festival

     in

     May

    ,

     is

     also

     a

     popular

     event

    .

     The

     city

     is

     also

     home

     to

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     several

     key

     trends

     that

     could

     influence

     the

     development

     and

     application

     of

     AI

     technologies

    .

     Here

     are

     some

     possible

     future

     trends

     that

     are

     likely

     to

     shape

     the

     development

     and

     application

     of

     AI

    :
    


    1

    .

     Advances

     in

     Computer

     Science

    :

     As

     computer

     science

     continues

     to

     advance

    ,

     the

     ability

     to

     build

     and

     operate

     powerful

     AI

     systems

     will

     likely

     increase

    .

     Advances

     in

     areas

     such

     as

     machine

     learning

    ,

     natural

     language

     processing

    ,

     and

     computer

     vision

     will

     allow

     AI

     systems

     to

     learn

     and

     adapt

     more

     quickly

     and

     accurately

     than

     ever

     before

    .
    


    2

    .

     Increased

     Use

     of

     AI

     in

     Healthcare

    :

     AI

     has

     the

     potential

     to

     revolution

    ize

     healthcare

     by

     providing

     faster

     and

     more

     accurate

     diagnoses

    ,

     personalized

     treatment

     plans

    ,

     and

     improved

     patient

     outcomes

    



```python
llm.shutdown()
```
