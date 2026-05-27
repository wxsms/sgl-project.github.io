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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.72it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.71it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:47,  3.99s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:47,  3.99s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:47,  3.99s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:47,  3.99s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:47,  3.99s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.62it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.81it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.81it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.81it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.81it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.81it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.81it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.81it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.81it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.81it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:04<00:03,  9.52it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:04<00:03,  9.52it/s]

    Compiling num tokens (num_tokens=512):  34%|███▍      | 20/58 [00:04<00:03,  9.52it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=240):  50%|█████     | 29/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=224):  50%|█████     | 29/58 [00:04<00:01, 15.99it/s]Compiling num tokens (num_tokens=224):  67%|██████▋   | 39/58 [00:04<00:00, 24.58it/s]Compiling num tokens (num_tokens=208):  67%|██████▋   | 39/58 [00:04<00:00, 24.58it/s]Compiling num tokens (num_tokens=192):  67%|██████▋   | 39/58 [00:04<00:00, 24.58it/s]Compiling num tokens (num_tokens=176):  67%|██████▋   | 39/58 [00:04<00:00, 24.58it/s]Compiling num tokens (num_tokens=160):  67%|██████▋   | 39/58 [00:04<00:00, 24.58it/s]Compiling num tokens (num_tokens=144):  67%|██████▋   | 39/58 [00:04<00:00, 24.58it/s]Compiling num tokens (num_tokens=128):  67%|██████▋   | 39/58 [00:04<00:00, 24.58it/s]Compiling num tokens (num_tokens=112):  67%|██████▋   | 39/58 [00:04<00:00, 24.58it/s]Compiling num tokens (num_tokens=96):  67%|██████▋   | 39/58 [00:04<00:00, 24.58it/s] 

    Compiling num tokens (num_tokens=80):  67%|██████▋   | 39/58 [00:04<00:00, 24.58it/s]Compiling num tokens (num_tokens=64):  67%|██████▋   | 39/58 [00:04<00:00, 24.58it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:04<00:00, 34.10it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:04<00:00, 34.10it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.33it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=56.03 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.00 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=56.00 GB):   3%|▎         | 2/58 [00:00<00:02, 19.33it/s]Capturing num tokens (num_tokens=7168 avail_mem=55.99 GB):   3%|▎         | 2/58 [00:00<00:02, 19.33it/s]Capturing num tokens (num_tokens=6656 avail_mem=55.99 GB):   3%|▎         | 2/58 [00:00<00:02, 19.33it/s]Capturing num tokens (num_tokens=6144 avail_mem=55.99 GB):   3%|▎         | 2/58 [00:00<00:02, 19.33it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=55.99 GB):   9%|▊         | 5/58 [00:00<00:02, 22.70it/s]Capturing num tokens (num_tokens=5632 avail_mem=55.98 GB):   9%|▊         | 5/58 [00:00<00:02, 22.70it/s]Capturing num tokens (num_tokens=5120 avail_mem=55.98 GB):   9%|▊         | 5/58 [00:00<00:02, 22.70it/s]Capturing num tokens (num_tokens=4608 avail_mem=55.97 GB):   9%|▊         | 5/58 [00:00<00:02, 22.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.97 GB):   9%|▊         | 5/58 [00:00<00:02, 22.70it/s]Capturing num tokens (num_tokens=4096 avail_mem=55.97 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.73it/s]Capturing num tokens (num_tokens=3840 avail_mem=55.97 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.73it/s]Capturing num tokens (num_tokens=3584 avail_mem=55.96 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.73it/s]Capturing num tokens (num_tokens=3328 avail_mem=55.96 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.73it/s]Capturing num tokens (num_tokens=3072 avail_mem=55.96 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.73it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=55.96 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.73it/s]Capturing num tokens (num_tokens=2816 avail_mem=55.96 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.72it/s]Capturing num tokens (num_tokens=2560 avail_mem=55.95 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.72it/s]Capturing num tokens (num_tokens=2304 avail_mem=55.95 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.72it/s]Capturing num tokens (num_tokens=2048 avail_mem=55.94 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.72it/s]Capturing num tokens (num_tokens=1792 avail_mem=55.94 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.94 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.72it/s]Capturing num tokens (num_tokens=1536 avail_mem=55.94 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=55.94 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.93it/s]Capturing num tokens (num_tokens=1024 avail_mem=55.92 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.93it/s]Capturing num tokens (num_tokens=960 avail_mem=55.93 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.93it/s] Capturing num tokens (num_tokens=896 avail_mem=55.93 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.93it/s]

    Capturing num tokens (num_tokens=832 avail_mem=55.92 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.93it/s]Capturing num tokens (num_tokens=768 avail_mem=55.92 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.93it/s]Capturing num tokens (num_tokens=768 avail_mem=55.92 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=704 avail_mem=55.92 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=640 avail_mem=55.91 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=576 avail_mem=55.91 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=512 avail_mem=55.90 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=480 avail_mem=55.91 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=448 avail_mem=55.91 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.07it/s]Capturing num tokens (num_tokens=448 avail_mem=55.91 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.60it/s]Capturing num tokens (num_tokens=416 avail_mem=55.91 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.60it/s]Capturing num tokens (num_tokens=384 avail_mem=55.91 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.60it/s]Capturing num tokens (num_tokens=352 avail_mem=55.90 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.60it/s]

    Capturing num tokens (num_tokens=320 avail_mem=55.90 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.60it/s]Capturing num tokens (num_tokens=288 avail_mem=55.90 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.60it/s]Capturing num tokens (num_tokens=256 avail_mem=55.89 GB):  53%|█████▎    | 31/58 [00:00<00:00, 45.60it/s]Capturing num tokens (num_tokens=256 avail_mem=55.89 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.15it/s]Capturing num tokens (num_tokens=240 avail_mem=55.89 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.15it/s]Capturing num tokens (num_tokens=224 avail_mem=55.89 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.15it/s]Capturing num tokens (num_tokens=208 avail_mem=55.88 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.15it/s]Capturing num tokens (num_tokens=192 avail_mem=55.88 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.15it/s]Capturing num tokens (num_tokens=176 avail_mem=55.88 GB):  64%|██████▍   | 37/58 [00:00<00:00, 47.15it/s]Capturing num tokens (num_tokens=160 avail_mem=55.88 GB):  64%|██████▍   | 37/58 [00:01<00:00, 47.15it/s]Capturing num tokens (num_tokens=160 avail_mem=55.88 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.10it/s]Capturing num tokens (num_tokens=144 avail_mem=55.87 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.10it/s]Capturing num tokens (num_tokens=128 avail_mem=55.87 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.10it/s]

    Capturing num tokens (num_tokens=112 avail_mem=55.87 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.10it/s]Capturing num tokens (num_tokens=96 avail_mem=55.86 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.10it/s] Capturing num tokens (num_tokens=80 avail_mem=55.86 GB):  74%|███████▍  | 43/58 [00:01<00:00, 48.10it/s]

    Capturing num tokens (num_tokens=80 avail_mem=55.86 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.91it/s]Capturing num tokens (num_tokens=64 avail_mem=55.86 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.91it/s]Capturing num tokens (num_tokens=48 avail_mem=55.85 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.91it/s]Capturing num tokens (num_tokens=32 avail_mem=55.85 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.91it/s]Capturing num tokens (num_tokens=28 avail_mem=55.84 GB):  83%|████████▎ | 48/58 [00:01<00:00, 33.91it/s]

    Capturing num tokens (num_tokens=28 avail_mem=55.84 GB):  90%|████████▉ | 52/58 [00:01<00:00, 25.66it/s]Capturing num tokens (num_tokens=24 avail_mem=55.84 GB):  90%|████████▉ | 52/58 [00:01<00:00, 25.66it/s]Capturing num tokens (num_tokens=20 avail_mem=55.84 GB):  90%|████████▉ | 52/58 [00:01<00:00, 25.66it/s]Capturing num tokens (num_tokens=16 avail_mem=54.74 GB):  90%|████████▉ | 52/58 [00:01<00:00, 25.66it/s]

    Capturing num tokens (num_tokens=12 avail_mem=54.73 GB):  90%|████████▉ | 52/58 [00:01<00:00, 25.66it/s]Capturing num tokens (num_tokens=12 avail_mem=54.73 GB):  97%|█████████▋| 56/58 [00:01<00:00, 20.49it/s]Capturing num tokens (num_tokens=8 avail_mem=54.73 GB):  97%|█████████▋| 56/58 [00:01<00:00, 20.49it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=54.72 GB):  97%|█████████▋| 56/58 [00:01<00:00, 20.49it/s]Capturing num tokens (num_tokens=4 avail_mem=54.72 GB): 100%|██████████| 58/58 [00:02<00:00, 28.88it/s]


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
    Generated text:  Tom, and I'm a computer programmer. I've been coding for years and love to see how other people code. I'm a little bit proud of how I've helped other people develop their skills, and I'm really excited to see what the future holds for me.
    The other person who is reading this text is Jack, a software engineer. Jack is looking to develop a new feature for a software project.
    Based on the information provided, what are some potential strategies for Jack to develop a new feature for the software project? To assist you in your development process, please provide some guidance on the best practices for software development, including coding
    ===============================
    Prompt: The president of the United States is
    Generated text:  a man, and the president of a country is also a man. Therefore, the president of the United States is also a man. The reasoning above is an example of which logical fallacy?
    A: Ad Hominem
    B: Hasty Generalization
    C: Affirming the Consequent
    D: False Dilemma
    
    To determine the type of logical fallacy in the given argument, let's break down the reasoning step by step.
    
    1. **Identify the type of logical fallacy:**
       The given argument is an example of a **false dilemma**.
    
    2. **Explanation of the fallacy:
    ===============================
    Prompt: The capital of France is
    Generated text:  ( )____
    A. Paris
    B. Marseille
    C. Lyon
    D. Toulouse
    Answer:
    
    A
    
    Which of the following statements about the structure and function of cells is incorrect?
    A. The cell membrane has a certain degree of flexibility to move against the flow of substances.
    B. The nuclear envelope is separated by a nuclear membrane, but the nuclear membrane itself is not a closed membrane.
    C. Both the cell membrane and nuclear membrane contain phospholipid molecules and protein molecules.
    D. The cytoplasm is a living environment for all living cells, and it can be considered a component of the cell membrane.
    
    ===============================
    Prompt: The future of AI is
    Generated text:  rapidly approaching. It is expected to revolutionize the way we live, work, and interact with the world. However, this rapid advancement also presents challenges that we need to address to ensure that AI is used in a responsible and ethical manner.
    
    AI is being used in a variety of ways, including in healthcare, finance, transportation, and education. These applications are driving the adoption of AI in a number of industries, but they also raise concerns about privacy, bias, and transparency. In this post, we will explore the importance of privacy and bias in AI and what steps can be taken to address these issues.
    
    Privacy concerns are one of the


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


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and vibrant cultural scene. It is also the birthplace of French literature, art, and cuisine. Paris is a city of contrasts, with its rich history and modernity intertwined. The city is home to the Louvre Museum, the Eiffel Tower, and the Notre-Dame Cathedral, among many other attractions. Paris is a city of art, culture, and history, and is a must-visit destination for anyone interested in French culture and history. 
    
    The city of Paris is a vibrant and dynamic place, with a rich history and a modern sense
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we interact with technology and the world around us. Here are some of the most likely trends that are likely to shape the future of AI:
    
    1. Increased automation and robotics: As AI technology continues to advance, we are likely to see an increase in automation and robotics in various industries. This will lead to the development of more efficient and cost-effective solutions that can perform tasks that were previously done by humans.
    
    2. AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we are likely
    


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
    Generated text:  [insert your name] and I am a friendly and helpful AI assistant who can assist with any questions or tasks you may have. I am always here to help, whether you need advice on [insert one or more relevant topic] or help with [insert one or more relevant project]. I am always ready to assist you in any way possible. So if you have any questions, concerns, or just want to chat, please feel free to reach out to me. I look forward to assisting you! 🤝✨🤗
    
    Hi there! I'm a friendly AI assistant here on [insert fictional platform name]. I'm here to help
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, which is the largest city and the country's political, economic, and cultural center.
    
    That's correct. Paris is the largest city in France and serves as the country's political, economic, and cultural center. The city is home to many famous landmarks and attractions, including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also known for its rich history, vibrant arts scene, and annual festivals and celebrations. Overall, Paris is a major city that has played a significant role in shaping French culture and identity. 
    
    This statement is accurate based on the information provided in the given answer. Paris is
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  set to be very exciting and diverse. Here are some potential trends to look out for:
    
    1. Increased Personalization: As AI gets more advanced, it will be able to learn and adapt to different people and situations. This means that AI will become more personal, offering better and more tailored experiences for users.
    
    2. Autonomous and Self-Driving Vehicles: Autonomous and self-driving vehicles are likely to become more common in the future. They will allow people to travel more easily and efficiently, and reduce the risk of accidents caused by human error.
    
    3. Integration with Human-Centered Design: AI will be more integrated with human-centered design to enhance


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

    Character

    's

     Name

    ],

     and

     I

    'm

     a

     [

    Nature

     or

     Profession

    ]

     with

     a

     passion

     for

     [

    Occup

    ation

    ].

     I

    'm

     a

     friendly

    ,

     knowledgeable

    ,

     and

     versatile

     individual

     who

     thr

    ives

     on

     sharing

     my

     experiences

     and

     insights

     with

     others

    .

     I

     believe

     that

     understanding

     the

     world

     around

     us

     can

     lead

     to

     personal

     growth

     and

     a

     deeper

     connection

     with

     oneself

     and

     others

    .

     What

     brings

     you

     to

     this

     place

    ?

     How

     did

     you

     get

     here

    ?

     What

     do

     you

     hope

     to

     achieve

     in

     your

     journey

    ?

     I

     look

     forward

     to

     having

     the

     opportunity

     to

     share

     more

     about

     myself

     and

     our

     shared

     interests

    .

     Thank

     you

    .

     
    


    Is

     this

     a

     nice

     introduction

    ?

     Do

     I

     need

     to

     add

     more

     details

    ?

     Yes

    ,

     you

     can

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     city

     of

     light

     and

     vibrant

     culture

    ,

     known

     for

     its

     iconic

     landmarks

     such

     as

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

     and

     Notre

    -D

    ame

     Cathedral

    .

     It

     is

     also

     home

     to

     numerous

     museums

    ,

     art

     galleries

    ,

     and

     festivals

    ,

     making

     it

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

     food

    ,

     cuisine

    ,

     and

     fashion

    ,

     with

     its

     iconic

     cafes

     and

     restaurants

    ,

     such

     as

     Le

     Bon

     March

    é

     and

     Le

     Com

    pt

    oir

     de

     la

     partie

    .

     It

     is

     also

     home

     to

     the

     European

     Parliament

    ,

     the

     headquarters

     of

     the

     European

     Union

    ,

     making

     it

     a

     center

     for

     political

     and

     cultural

     events

    .

     Overall

    ,

     Paris

     is

     a

     city

     of

     history

    ,

     culture

    ,

     and

     vibrant

     energy

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     likely

     to

     be

     characterized

     by

     more

     integration

     with

     human

     intelligence

    ,

     leading

     to

     more

     natural

     and

     intelligent

     interactions

    .

     In

     the

     near

     future

    ,

     we

     can

     expect

     more

     advanced

     AI

     systems

     to

     be

     able

     to

     understand

     and

     interpret

     human

     emotions

    ,

     and

     to

     learn

     and

     improve

     on

     a

     continuous

     basis

    .

     There

     will

     be

     a

     greater

     emphasis

     on

     creating

     AI

     that

     is

     both

     powerful

     and

     ethical

    ,

     with

     an

     increased

     focus

     on

     privacy

    ,

     security

    ,

     and

     accountability

     in

     the

     development

     and

     use

     of

     AI

    .

     Additionally

    ,

     there

     will

     likely

     be

     continued

     development

     of

     AI

     that

     can

     adapt

     to

     new

     challenges

     and

     situations

    ,

     and

     that

     can

     collaborate

     and

     integrate

     with

     other

     technologies

     and

     systems

     in

     new

     ways

    .

     Overall

    ,

     the

     future

     of

     AI

     is

    



```python
llm.shutdown()
```
