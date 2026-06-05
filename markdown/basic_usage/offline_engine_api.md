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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.45it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.44it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:29,  5.78s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:29,  5.78s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:29,  5.78s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:29,  5.78s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:05<05:29,  5.78s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]

    Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:05<00:47,  1.13it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:15,  3.07it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:15,  3.07it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:15,  3.07it/s]Compiling num tokens (num_tokens=2816):  19%|█▉        | 11/58 [00:06<00:15,  3.07it/s]Compiling num tokens (num_tokens=2560):  19%|█▉        | 11/58 [00:06<00:15,  3.07it/s]Compiling num tokens (num_tokens=2304):  19%|█▉        | 11/58 [00:06<00:15,  3.07it/s]Compiling num tokens (num_tokens=2048):  19%|█▉        | 11/58 [00:06<00:15,  3.07it/s]Compiling num tokens (num_tokens=1792):  19%|█▉        | 11/58 [00:06<00:15,  3.07it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:06<00:06,  6.08it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:06<00:06,  6.08it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:06<00:06,  6.08it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:06<00:06,  6.08it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:06<00:06,  6.08it/s] 

    Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:06<00:06,  6.08it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:06<00:06,  6.08it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:06<00:03,  9.28it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:06<00:03,  9.28it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:06<00:03,  9.28it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:06<00:03,  9.28it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:06<00:03,  9.28it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:06<00:03,  9.28it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:06<00:03,  9.28it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:06<00:02, 13.12it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:06<00:02, 13.12it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:06<00:02, 13.12it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:06<00:02, 13.12it/s]

    Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:06<00:02, 13.12it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:06<00:02, 13.12it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:06<00:02, 13.12it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:06<00:01, 17.80it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:06<00:01, 17.80it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:06<00:01, 17.80it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:06<00:01, 17.80it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:06<00:01, 17.80it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:06<00:01, 17.80it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:06<00:01, 17.80it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:06<00:01, 17.80it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:06<00:01, 17.80it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 25.48it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 25.48it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 25.48it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 25.48it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:06<00:00, 25.48it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:06<00:00, 25.48it/s]

    Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:06<00:00, 25.48it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:06<00:00, 25.48it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:06<00:00, 25.48it/s]Compiling num tokens (num_tokens=24):  76%|███████▌  | 44/58 [00:06<00:00, 25.48it/s]Compiling num tokens (num_tokens=20):  76%|███████▌  | 44/58 [00:06<00:00, 25.48it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:06<00:00, 36.39it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:06<00:00, 36.39it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:06<00:00, 36.39it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:06<00:00, 36.39it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:06<00:00, 36.39it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.68it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 17.40it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:03, 17.40it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 17.40it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:03, 17.40it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.57it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 20.57it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 20.57it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 20.57it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.06it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.06it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.06it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  14%|█▍        | 8/58 [00:00<00:02, 24.06it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  21%|██        | 12/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  21%|██        | 12/58 [00:00<00:01, 28.35it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.85it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.85it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  29%|██▉       | 17/58 [00:00<00:01, 33.85it/s] Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.50it/s]Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.50it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.50it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.50it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.50it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  38%|███▊      | 22/58 [00:00<00:00, 37.50it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.77it/s]

    Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  47%|████▋     | 27/58 [00:00<00:00, 39.77it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  55%|█████▌    | 32/58 [00:00<00:00, 41.56it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.21it/s]

    Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  64%|██████▍   | 37/58 [00:01<00:00, 43.21it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.51it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  72%|███████▏  | 42/58 [00:01<00:00, 44.51it/s] Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  81%|████████  | 47/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 44.33it/s]

    Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  81%|████████  | 47/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  81%|████████  | 47/58 [00:01<00:00, 44.33it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  90%|████████▉ | 52/58 [00:01<00:00, 44.68it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.96it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 44.96it/s]

    Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 39.15it/s]


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
    Generated text:  Tommy, a rookie programmer who has just started to learn how to code. I'm currently working on a new project that involves creating a simple calculator program. Can you explain to me how to create a basic calculator program in Python that performs addition, subtraction, multiplication, and division? Also, could you provide some examples of how to test the program with a range of values and error handling? Additionally, I would like to know if you have any tips on how to optimize the program to handle large numbers efficiently. Lastly, could you help me write a simple function that multiplies two numbers and returns the result? To make the code even
    ===============================
    Prompt: The president of the United States is
    Generated text:  a type of ____.
    A. Monarch
    B. Head of State
    C. Head of Government
    D. Premier
    Answer: B
    
    According to the provisions of the "Contract Law of the People's Republic of China", where one party provides a guarantee with the intention to be compensated but fails to do so within ____ after becoming aware of the occurrence of the debt, the other party may exercise the right of subrogation against the guarantor.
    A. 1 month
    B. 2 months
    C. 3 months
    D. 6 months
    Answer: C
    
    For a certain construction project, the engineering
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The Capital of India is Delhi. Which of the two capitals are located in the northern part of the country? To determine which capital of France and India is located in the northern part of the country, we need to consider the geographical locations of these capitals.
    
    1. **Capital of France**: Paris
    2. **Capital of India**: Delhi
    
    Let's analyze each capital's position:
    
    - **Paris**: Located in the northwestern region of France, bordering the English Channel.
    - **Delhi**: Located in the northern region of India, bordering the Indian Ocean.
    
    The northern part of the country is typically characterized by its
    ===============================
    Prompt: The future of AI is
    Generated text:  about many things but one of the most important aspects of it, if not the most important, is the ethical implications. The great example of this is the situation of the US President Donald Trump, who has made it clear that he wants to make America the first country to fully implement and use AI. So far, the American government is one of the leaders in the field of AI, leading the way in the area of artificial intelligence. Many are working hard to implement AI in the public sector, especially in healthcare and manufacturing. The need for AI in the US is not only due to the demand of the job market but also due to the


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


    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. It is also a major center for business and finance, with many multinational corporations headquartered there. The city is known for its fashion industry, with Paris Fashion Week being one of the world's largest fashion events. The French capital is a vibrant and dynamic city with a rich history and culture. It is a popular tourist destination and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the development of the technology in the coming years. Here are some of the most likely trends:
    
    1. Increased integration with human intelligence: As AI becomes more advanced, it is likely to become more integrated with human intelligence. This means that AI systems will be able to learn from and adapt to human behavior, making them more capable of understanding and responding to human emotions and motivations.
    
    2. Greater emphasis on ethical considerations: As AI becomes more advanced, there will be a greater emphasis on ethical considerations. This means that AI systems will be designed to be transparent, accountable, and
    


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
    Generated text:  [Your Name], and I am a [职业] with experience in [职前职位] at [公司名称]. I believe I am well-qualified to take on this role, and I look forward to contributing to your team.
    
    **Your Name:** [Your Full Name]
    **Date:** [Today's Date]
    **Position:** [Your Job Title]
    **Company:** [Company Name]
    ---
    
    Remember, this is a neutral self-introduction, and it is open to any character or role. Adjust the content and style to match the needs of your setting or characters. Good luck! 🎉✨
    
    ---
    
    This is a great
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    
    Which of the following sentences is grammatically incorrect?
    
    A) The weather will be cool tomorrow.
    B) She is eager to go to France.
    C) The book is free for 15 cents.
    D) The temperature will be 75 degrees F.
    
    The sentence that is grammatically incorrect is:
    
    D) The temperature will be 75 degrees F.
    
    This sentence has a grammatical error in the comma placement. The correct sentence should be:
    
    D) The temperature will be 75 degrees Fahrenheit. 
    
    The correct answer is D. The sentence with the grammatical error is D) The temperature will be 
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be one of rapid and dramatic change. Here are some possible trends in AI in the coming years:
    
    1. AI will continue to improve its accuracy and efficiency, but it will also become more conscious and self-aware. This will involve developing AI systems that can learn from experience, adapt to changing conditions, and make decisions based on their own goals and objectives.
    
    2. AI will become more integrated with other technologies and services, such as sensors, biometric data, and social media. This will allow for more connected and personalized experiences, as well as improved security and privacy.
    
    3. AI will also become more ethical and responsible. As


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

    name

    ],

     and

     I

     am

     a

     [

    description

     of

     your

     profession

    ].

     I

     enjoy

     [

    reason

     for

     interest

     in

     your

     profession

    ].

     In

     my

     free

     time

    ,

     I

     like

     [

    activity

     that

     I

     enjoy

    ].

     I

     am

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

    .

     How

     can

     I

     get

     to

     know

     you

     better

    ?


    Hello

     [

    name

    ],

     my

     name

     is

     [

    name

    ]

     and

     I

     am

     a

     [

    description

     of

     your

     profession

    ].

     I

     enjoy

     [

    reason

     for

     interest

     in

     your

     profession

    ].

     In

     my

     free

     time

    ,

     I

     like

     [

    activity

     that

     I

     enjoy

    ].

     I

     am

     excited

     to

     meet

     you

     and

     learn

     more

     about

     you

    .

     How

     can

     I

     get

     to

     know

     you

     better

    ?

     What

     would

     you

     like

     to

     know

     about

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Saint

    e

    -

    Port

    e

    ".

     It

     is

     the

     largest

     city

     in

     France

    ,

     and

     is

     home

     to

     many

     of

     France

    's

     cultural

    ,

     political

    ,

     and

     economic

     institutions

    ,

     including

     the

     headquarters

     of

     major

     French

     companies

     and

     the

     French

     Parliament

    .

     It

     is

     also

     the

     largest

     city

     in

     the

     European

     Union

    ,

     home

     to

     many

     of

     Europe

    's

     most

     prominent

     museums

     and

     attractions

    .

     The

     city

     has

     a

     rich

     history

     and

     is

     known

     for

     its

     art

    ,

     music

    ,

     and

     cuisine

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

     many

     narrow

     streets

    ,

     medieval

     architecture

    ,

     and

     vibrant

     nightlife

    .

     Its

     status

     as

     a

     major

     city

     has

     brought

     many

     changes

     to

     the

     city

     over

     the

     centuries

    ,

     including

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     full

     of

     exciting

     developments

     and

     innovations

    ,

     but

     some

     trends

     that

     are

     likely

     to

     shape

     the

     landscape

     of

     AI

     in

     the

     coming

     years

     include

    :
    


    1

    .

     Increased

     AI

     ethics

    :

     As

     AI

     becomes

     more

     integrated

     into

     daily

     life

    ,

     the

     need

     for

     ethical

     guidelines

     and

     standards

     will

     become

     more

     critical

    .

     We

     may

     see

     new

     ethical

     frameworks

     being

     developed

     to

     govern

     AI

     use

    ,

     with

     considerations

     of

     fairness

    ,

     accountability

    ,

     and

     transparency

    .
    


    2

    .

     Improved

     AI

     algorithms

    :

     AI

     algorithms

     will

     become

     even

     more

     sophisticated

     and

     capable

    ,

     capable

     of

     handling

     larger

     and

     more

     complex

     data

     sets

    .

     This

     will

     require

     continued

     improvements

     in

     machine

     learning

     and

     natural

     language

     processing

    .
    


    3

    .

     Increased

     AI

     integration

     with

     human

     AI

    :

     As

     AI

    



```python
llm.shutdown()
```
