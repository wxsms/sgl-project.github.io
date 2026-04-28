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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.89it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.88it/s]


    2026-04-28 21:46:26,738 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-28 21:46:26] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<04:10,  4.39s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:35,  1.48it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:10,  4.42it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:10,  4.42it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]

    Compiling num tokens (num_tokens=416):  38%|███▊      | 22/58 [00:04<00:03,  9.94it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=224):  55%|█████▌    | 32/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=208):  55%|█████▌    | 32/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=192):  55%|█████▌    | 32/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=176):  55%|█████▌    | 32/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=160):  55%|█████▌    | 32/58 [00:04<00:01, 16.72it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:04<00:00, 25.67it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:04<00:00, 25.67it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:04<00:00, 25.67it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:04<00:00, 25.67it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:04<00:00, 25.67it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:04<00:00, 25.67it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:04<00:00, 25.67it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:04<00:00, 25.67it/s]Compiling num tokens (num_tokens=32):  74%|███████▍  | 43/58 [00:04<00:00, 25.67it/s]Compiling num tokens (num_tokens=28):  74%|███████▍  | 43/58 [00:04<00:00, 25.67it/s]

    Compiling num tokens (num_tokens=24):  74%|███████▍  | 43/58 [00:05<00:00, 25.67it/s]Compiling num tokens (num_tokens=20):  74%|███████▍  | 43/58 [00:05<00:00, 25.67it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:05<00:00, 35.73it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:05<00:00, 35.73it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:05<00:00, 35.73it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:05<00:00, 35.73it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:05<00:00, 35.73it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.49it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 20.00it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 20.00it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 20.00it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 20.00it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 23.34it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 23.34it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 23.34it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 23.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 23.34it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.47it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.47it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.47it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.47it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.47it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 28.47it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.12it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.02it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.02it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.02it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.02it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.02it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.02it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.02it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  43%|████▎     | 25/58 [00:00<00:00, 43.22it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.15it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.15it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.15it/s]Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.15it/s]

    Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.15it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.15it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  53%|█████▎    | 31/58 [00:00<00:00, 46.15it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.54it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.54it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.54it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.54it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.54it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.54it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  64%|██████▍   | 37/58 [00:00<00:00, 48.54it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.98it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.98it/s]Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.98it/s]

    Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.98it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.98it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.98it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  74%|███████▍  | 43/58 [00:01<00:00, 49.98it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.57it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.57it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.57it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.57it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.57it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.57it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  84%|████████▍ | 49/58 [00:01<00:00, 50.57it/s]Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  95%|█████████▍| 55/58 [00:01<00:00, 50.99it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  95%|█████████▍| 55/58 [00:01<00:00, 50.99it/s]

    Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  95%|█████████▍| 55/58 [00:01<00:00, 50.99it/s] Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  95%|█████████▍| 55/58 [00:01<00:00, 50.99it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 44.41it/s]


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
    Generated text:  Anna, a graduate student in the Department of Pharmacology and Toxicology at the University of California, Irvine. I am a theoretical chemist by training, and my research focuses on the development of materials and interfaces that can improve drug delivery and biosensing capabilities. My research has been supported by the National Institutes of Health, the National Science Foundation, the Department of Energy, and the European Union.
    I am a member of the Computational Chemistry and Bioconjugate Chemistry Groups of the UC Irvine Chemistry Department, and a member of the Biomaterials Group of the UC Irvine Materials Science and Engineering Department. I am also a member of the Center
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political office that holds great potential for growth. In the 19th century, the United States had three branches of government, which are the executive, legislative, and judicial branches. The executive branch is the branch that is responsible for the executive branch's role, including the appointment of federal judges, managing the executive branch, and handling foreign affairs. The legislative branch is responsible for enacting laws that define the scope of the executive branch's powers. The judicial branch is responsible for determining the validity of laws and making sure that the executive branch does not infringe on certain rights. The president's ability to be elected to the office of
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. 正确
    B. 错误
    Answer:
    
    A
    
    In the event of a fire in a building, which of the following should be the first action to take?
    A. Notify all residents
    B. Evacuate all occupants
    C. Request fire service assistance
    D. Call the fire department
    Answer:
    
    B
    
    Which of the following statements about the decision-making process for macroeconomic regulation is correct?
    A. The decision-making process for macroeconomic regulation involves a complete cycle, including preparation, implementation, feedback, and evaluation.
    B. The decision-making process for macroeconomic regulation is constrained by the
    ===============================
    Prompt: The future of AI is
    Generated text:  in the present
    Artificial intelligence (AI) has changed our lives in countless ways. From the precision of healthcare to the efficiency of transportation, AI is revolutionizing the way we live our lives.
    But as the AI landscape expands, so does the need for experts with the skills to apply AI to solve complex problems and deliver results. That's where AI researchers like you come in.
    You bring a wealth of experience and knowledge to the table, and you can make a real difference in helping us unlock new opportunities and find practical solutions to the challenges we face.
    Your role as an AI researcher doesn't just involve being a data scientist or an


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


    Generated text:  [Name] and I'm a [Age] year old [Occupation]. I'm a [Type of Person] who enjoys [Favorite Activity/Interest]. I'm [Type of Person] and I'm always [Positive Trait]. I'm [Type of Person] and I'm always [Positive Trait]. I'm [Type of Person] and I'm always [Positive Trait]. I'm [Type of Person] and I'm always [Positive Trait]. I'm [Type of Person] and I'm always [Positive Trait]. I'm [Type of Person] and I'm always [Positive Trait]. I'm [Type of Person
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting the annual Eiffel Tower Festival and hosting the World Cup of football. Paris is a popular tourist destination, with millions of visitors annually. The city is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on the world. It is also a major center for science and technology, with many universities and research institutions. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: As AI becomes more sophisticated, it is likely to become more integrated with human intelligence, allowing it to learn and adapt to new situations more effectively. This could lead to more complex and nuanced AI systems that can better understand and respond to human emotions and behaviors.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI becomes more advanced, it is likely to be used in more complex and personalized ways, potentially leading to even more effective and efficient healthcare systems.
    
    3. Increased use of AI
    


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
    Generated text:  [insert name]. I am a [insert profession]. I have [insert relevant experience or skills]. I am [insert personal qualities or interests]. What can you tell me about yourself? Let me know if you have any questions. [Insert name] Hello, my name is [insert name]. I am a [insert profession]. I have [insert relevant experience or skills]. I am [insert personal qualities or interests]. What can you tell me about yourself? Let me know if you have any questions. [Insert name] How about you? What can you tell me about yourself? [Insert name] I am [insert name]. I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is a large and historic city located on the Île de la Cité, on the banks of the Seine River, which gives its name to the city. Paris is also home to the Eiffel Tower, the Louvre Museum, the Arc de Triomphe, and many other iconic structures and attractions. As one of the world's most cosmopolitan cities, Paris is a cultural and intellectual hub, known for its cafes, museums, and food. Its status as the political and economic capital of France also makes it a major center for government and business. Despite its size and history, Paris remains a very popular
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  highly promising, and it is projected to continue to grow and evolve rapidly in the coming years. Here are some potential trends in the field of AI:
    
    1. Increased focus on ethics and fairness: With the increasing concern about AI's impact on society and individual privacy, there is a push for more ethical and fair AI systems. AI developers are increasingly focusing on creating systems that are transparent, explainable, and have no bias.
    
    2. More integration with human expertise: As AI becomes more integrated with human expertise, it will become increasingly important to have human oversight and validation of AI decisions. This is particularly important in areas like healthcare, finance,


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

    _,

     and

     I

     am

     a

    /an

     ______

    __

    _.

     I

    'm

     a

    /an

     ______

    __

     with

     a

    /an

     ______

    __

     name

    .

     
    


    Describe

     your

     current

     location

    ,

     what

     you

     do

     for

     a

     living

    ,

     and

     any

     notable

     accomplishments

    .

     Also

    ,

     tell

     me

     about

     your

     hobbies

     and

     interests

    .

     Lastly

    ,

     what

     kind

     of

     personality

     are

     you

    ?

     I

     want

     to

     get

     to

     know

     you

     better

     through

     this

     introduction

    .

     Be

     as

     detailed

     as

     possible

    .

     
    


    Additional

     information

     about

     yourself

     would

     also

     be

     appreciated

    .

     
    


    If

     you

     have

     any

     questions

     about

     me

    ,

     please

     let

     me

     know

    .

     


    Thank

     you

     for

     taking

     the

     time

     to

     meet

     me

    .

     


    [

    Your

     Name

    ]

     
    


    [

    Your

     Address

    ]

      


    [

    City

    ,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     
    


    A

     city

     renowned

     for

     its

     architectural

     beauty

    ,

     Paris

     is

     the

     

    1

    7

    th

     largest

     city

     in

     the

     world

     by

     population

     and

     the

     

    3

    rd

     most

     populous

     city

     in

     Europe

    .

     It

     is

     also

     the

     world

    's

     most

     populous

     city

     and

     the

     seat

     of

     government

    ,

     the

     Lou

    vre

     Museum

    ,

     and

     of

     the

     Paris

     Mét

    ro

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

    ,

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

     and

     the

     Ch

    amps

    -

    É

    lys

    ées

    .

     The

     city

     is

     known

     for

     its

     food

    ,

     fashion

    ,

     music

    ,

     and

     arts

    ,

     and

     hosts

     many

     festivals

     throughout

     the

     year

    .

     
    


    Paris

     is

     the

     capital

     of

     France

     and

     serves

     as

     the

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     set

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     technological

     advancements

    ,

     changing

     consumer

     behavior

    ,

     and

     advances

     in

     machine

     learning

    .

     Here

     are

     some

     potential

     trends

     that

     could

     influence

     the

     future

     of

     artificial

     intelligence

    :
    


    1

    .

     Increased

     focus

     on

     ethics

     and

     safety

    :

     As

     more

     companies

     and

     governments

     invest

     in

     AI

    ,

     there

     will

     be

     a

     growing

     demand

     for

     ethical

     considerations

     and

     safeguards

    .

     Future

     AI

     systems

     will

     need

     to

     be

     designed

     to

     avoid

     causing

     harm

     to

     users

     and

     to

     ensure

     that

     they

     are

     safe

     and

     reliable

    .
    


    2

    .

     Aug

    mented

     reality

     and

     virtual

     reality

    :

     As

     AI

     becomes

     more

     integrated

     into

     our

     daily

     lives

    ,

     we

     are

     likely

     to

     see

     more

     immersive

     experiences

     that

     simulate

     real

    -world

     environments

    .

     Aug

    mented

     reality

    



```python
llm.shutdown()
```
