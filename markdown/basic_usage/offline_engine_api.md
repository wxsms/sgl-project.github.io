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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.52it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  1.52it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:45,  5.00s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:45,  5.00s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<04:45,  5.00s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<04:45,  5.00s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<04:45,  5.00s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:40,  1.30it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:05<00:13,  3.51it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:05<00:13,  3.51it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:05<00:13,  3.51it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:05<00:13,  3.51it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:05<00:13,  3.51it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:05<00:13,  3.51it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:05<00:13,  3.51it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:05<00:13,  3.51it/s]Compiling num tokens (num_tokens=1536):  19%|█▉        | 11/58 [00:05<00:13,  3.51it/s]Compiling num tokens (num_tokens=1280):  19%|█▉        | 11/58 [00:05<00:13,  3.51it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:05<00:04,  7.89it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=768):  34%|███▍      | 20/58 [00:05<00:04,  7.89it/s]

    Compiling num tokens (num_tokens=704):  34%|███▍      | 20/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=640):  34%|███▍      | 20/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=576):  34%|███▍      | 20/58 [00:05<00:04,  7.89it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:05<00:02, 12.67it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:05<00:01, 17.55it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:05<00:01, 17.55it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:05<00:01, 17.55it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:05<00:01, 17.55it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:05<00:01, 17.55it/s]

    Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:05<00:01, 17.55it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:05<00:01, 17.55it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:05<00:01, 17.55it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:05<00:01, 17.55it/s]Compiling num tokens (num_tokens=144):  60%|██████    | 35/58 [00:05<00:01, 17.55it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:05<00:00, 25.21it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:05<00:00, 25.21it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:05<00:00, 25.21it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:05<00:00, 25.21it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:05<00:00, 25.21it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:05<00:00, 25.21it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:05<00:00, 25.21it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:05<00:00, 25.21it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:05<00:00, 25.21it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:05<00:00, 32.44it/s]

    Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:05<00:00, 32.44it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:05<00:00, 32.44it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.93it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.74 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:04, 11.82it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:04, 11.82it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:04, 11.82it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   3%|▎         | 2/58 [00:00<00:04, 11.82it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.70 GB):   9%|▊         | 5/58 [00:00<00:03, 16.65it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.69 GB):   9%|▊         | 5/58 [00:00<00:03, 16.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:03, 16.65it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):   9%|▊         | 5/58 [00:00<00:03, 16.65it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.68 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.63it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.63it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.67 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.63it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.67 GB):  21%|██        | 12/58 [00:00<00:01, 25.76it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 25.76it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 25.76it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.66 GB):  21%|██        | 12/58 [00:00<00:01, 25.76it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 25.76it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  21%|██        | 12/58 [00:00<00:01, 25.76it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.92it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.65 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.92it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.92it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.92it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.62 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.92it/s]Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  29%|██▉       | 17/58 [00:00<00:01, 30.92it/s] Capturing num tokens (num_tokens=960 avail_mem=74.64 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=896 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=832 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=768 avail_mem=74.63 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.40it/s]

    Capturing num tokens (num_tokens=704 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  38%|███▊      | 22/58 [00:00<00:01, 34.40it/s]Capturing num tokens (num_tokens=640 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.48it/s]Capturing num tokens (num_tokens=576 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.48it/s]Capturing num tokens (num_tokens=512 avail_mem=74.60 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.48it/s]Capturing num tokens (num_tokens=480 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.48it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  47%|████▋     | 27/58 [00:00<00:00, 36.48it/s]Capturing num tokens (num_tokens=448 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=416 avail_mem=74.62 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=384 avail_mem=74.61 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.80it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.61 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  53%|█████▎    | 31/58 [00:01<00:00, 36.80it/s]Capturing num tokens (num_tokens=320 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=288 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=256 avail_mem=74.60 GB):  60%|██████    | 35/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=240 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=224 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  60%|██████    | 35/58 [00:01<00:00, 37.47it/s]Capturing num tokens (num_tokens=208 avail_mem=74.59 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.10it/s]Capturing num tokens (num_tokens=192 avail_mem=74.59 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.10it/s]Capturing num tokens (num_tokens=176 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.10it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.10it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.10it/s]Capturing num tokens (num_tokens=144 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=128 avail_mem=74.58 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=112 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=96 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.22it/s] Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  76%|███████▌  | 44/58 [00:01<00:00, 38.22it/s]Capturing num tokens (num_tokens=80 avail_mem=74.57 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.33it/s]Capturing num tokens (num_tokens=64 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.33it/s]Capturing num tokens (num_tokens=48 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.33it/s]

    Capturing num tokens (num_tokens=32 avail_mem=74.56 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.33it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  83%|████████▎ | 48/58 [00:01<00:00, 37.33it/s]Capturing num tokens (num_tokens=28 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=24 avail_mem=74.55 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=20 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=16 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  90%|████████▉ | 52/58 [00:01<00:00, 36.21it/s]Capturing num tokens (num_tokens=12 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.46it/s]Capturing num tokens (num_tokens=8 avail_mem=74.54 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.46it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=74.53 GB):  97%|█████████▋| 56/58 [00:01<00:00, 35.46it/s]Capturing num tokens (num_tokens=4 avail_mem=74.53 GB): 100%|██████████| 58/58 [00:01<00:00, 32.66it/s]


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
    Generated text:  Mika, I'm 24 years old, I have a job in an office. I love my job, but I also have a big family. Every day is a challenge for me. I have to make sure I'm taking care of both my family and my job. What do you think of my job, Mika? Please be specific.
    
    Mika's Answer: Well, I think the job you have is very fulfilling. I do get to work with people of various ages and experiences and learn a lot about different cultures and ideas. I'm also able to balance my work and family responsibilities, which I find valuable.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of which of the following organizations? A. Congress B. The White House C. The Supreme Court D. None of the above.
    Answer:
    
    B
    
    Patient: Ms. Li, 50 years old, has had bilateral foot pain for more than 2 years, and has had more than 40 episodes of urinary incontinence during the past 6 months. Physical examination: bilateral lower extremity edema, lumbar tenderness, hyperactive Babinski sign bilaterally. The most likely diagnosis is:
    A. Degenerative joint disease
    B. Peripheral neuropathy
    C. Lumbosacral
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It has a population of 2, 536, 073. It has a total area of 238 km². It has an altitude of 3, 376 m. In the North part of Paris there is a cathedral called the Notre Dame Cathedral. It has a total area of 68, 972 m². It has a population of 16, 350. It has a population of 7, 460. It has a population of 5, 320. It has a population of 4,
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, and it has the potential to change our world in profound ways. However, the rapid pace of technological advancement also presents a significant challenge, particularly in the areas of security and privacy. In this post, we will explore some of the ways in which AI is being used to protect our privacy and security.
    The first step in protecting our privacy is to educate ourselves about the types of data that are being collected and how they are being used. This includes understanding the potential risks associated with the use of AI and how it can be used to mislead or deceive individuals.
    The second step is to take steps to secure our data. This includes


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a brief description of your job or experience here]. I'm always looking for new challenges and opportunities to grow and learn. What are some of your favorite things to do? I love [insert a hobby or activity here]. I'm always looking for new experiences and adventures to try. What do you like to do for fun? I enjoy [insert a hobby or activity here]. I'm always looking for new ways to challenge myself and expand
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and festivals throughout the year. Paris is a popular tourist destination and a major hub for international business and diplomacy. The city is known for its rich history, diverse culture, and vibrant nightlife. It is a major transportation hub, with many major highways and rail lines connecting the city to other parts of France and the world. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another.
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some possible future trends in AI:
    
    1. Increased automation and artificial intelligence: As AI becomes more advanced, it is expected to automate more tasks, freeing up human workers to focus on more complex and creative work. This could lead to a shift in the job market, with many jobs being replaced by AI.
    
    2. Improved privacy and security: As AI systems become more sophisticated, there will be a need to address privacy and security concerns. This could involve developing new technologies to protect user data and ensuring that
    


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
    Generated text:  [Name] and I'm a [Age] year old [Sex]. I enjoy [Favorite Thing to Do], [Favorite Place], or [Favorite Book]. I am [特长/能力] and I have a passion for [My Passion for], [My Passion for]. I am always looking for [What I'm Looking For in a Person], [What I'm Looking For in a Job], or [What I'm Looking For in a Mentor]. If you have any questions or would like to know more about me, feel free to ask and I'm always here to answer. Please let me know what you would like to call me
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    A quick and detailed answer about the capital city of France would be:
    
    The capital of France is Paris, which is the capital of France. It's the largest city in Europe, located in the region of Hauts-de-France. Paris is home to many of Europe's top museums and monuments, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Notre-Dame Basilica. The city is also famous for its cuisine, fashion, and architecture. Paris is known for its lively nightlife and vibrant culture. The French Parliament building, known as the Eiffel Tower, is one of the
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by a number of different trends, including:
    
    1. Increased use of AI in healthcare: AI is being used to improve diagnosis and treatment of medical conditions, and to personalize treatment plans for patients. For example, AI can analyze medical images to detect cancer and other diseases, and can provide personalized medication recommendations based on a patient's genetic makeup.
    
    2. Integration of AI into transportation: As AI becomes more advanced, it is likely to be used in the transportation industry, including autonomous vehicles and ride-sharing services. These technologies could lead to more efficient, safe, and cost-effective transportation options for people and businesses alike.
    
    3.


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

    .

     I

    'm

     a

    /an

     __

    ________

    __

    _.

     I

     have

     a

    /an

     __

    ________

    _

     background

    .

     What

     do

     you

     think

     you

     can

     contribute

     to

     our

     team

    ?

     What

     is

     your

     area

     of

     expertise

     or

     strength

     that

     makes

     you

     unique

    ?


    I

    'm

     __

    ________

    _

     because

     I

     have

     __

    ________

    _

    .


    I

    'm

     __

    ________

    _

     because

     __

    ________

    _

    .


    What

     kind

     of

     experience

     do

     you

     have

     that

     you

     believe

     will

     be

     beneficial

     for

     our

     team

    ?

     I

    'm

     __

    ________

    _

     because

     __

    ________

    _

    .


    And

    ,

     finally

    ,

     what

     interests

     me

     about

     your

     team

    ?

     I

    'm

     __

    ________

    _

     because

     __

    ________

    _

    .


    I

     would

     like

     to

     ask

     you

     __

    ________

    _

     to

     get

     to

     know

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     city

     of

     light

    .


    Paris

     is

     the

     largest

     city

     in

     France

     and

     a

     UNESCO

     World

     Heritage

     Site

    .

     It

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

    ,

     Notre

     Dame

     Cathedral

    ,

     and

     the

     Lou

    vre

     Museum

    .

     The

     city

     is

     also

     known

     for

     its

     rich

     cultural

     heritage

     and

     lively

     street

     life

    ,

     with

     many

     cafes

    ,

     bars

    ,

     and

     restaurants

     serving

     up

     to

     

    3

    0

    0

    ,

    0

    0

    0

     visitors

     annually

    .

     Despite

     its

     size

    ,

     Paris

     remains

     a

     popular

     tourist

     destination

     for

     its

     beautiful

     architecture

    ,

     beautiful

     views

    ,

     and

     exciting

     culture

    .

     Paris

     was

     the

     capital

     of

     France

     until

     

    1

    7

    9

    0

     when

     the

     monarchy

     was

     abolished

    ,

     and

     it

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     fascinating

    ,

     and

     it

    's

     a

     rapidly

     evolving

     field

    .

     Some

     possible

     trends

     in

     AI

     include

    :
    


    1

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

     more

     integrated

     with

     other

     technologies

    ,

     such

     as

     blockchain

    ,

     quantum

     computing

    ,

     and

     machine

     learning

    .

     This

     will

     likely

     lead

     to

     new

     applications

     and

     developments

     in

     the

     field

    .
    


    2

    .

     Personal

    ization

     and

     relevance

    :

     AI

     is

     becoming

     more

     personalized

     and

     relevant

     to

     individuals

    ,

     which

     could

     lead

     to

     new

     applications

     in

     areas

     such

     as

     healthcare

    ,

     finance

    ,

     and

     education

    .
    


    3

    .

     Autonomous

     vehicles

    :

     With

     the

     increasing

     focus

     on

     autonomous

     vehicles

    ,

     AI

     is

     likely

     to

     become

     a

     more

     prominent

     force

     in

     the

     technology

     industry

    .
    


    4

    .

     Eth

    ical

     concerns

    :

     As

     AI

     technology

    



```python
llm.shutdown()
```
