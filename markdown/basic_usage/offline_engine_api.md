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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.98it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.98it/s]


    2026-05-04 21:05:19,278 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-04 21:05:19] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<03:56,  4.14s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<03:56,  4.14s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<03:56,  4.14s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=5632):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]Compiling num tokens (num_tokens=5120):   5%|▌         | 3/58 [00:04<01:01,  1.12s/it]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=4096):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=3840):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=3584):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=3328):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=3072):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=2816):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=2560):  12%|█▏        | 7/58 [00:04<00:19,  2.65it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:04<00:05,  7.30it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:04<00:05,  7.30it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:04<00:05,  7.30it/s]Compiling num tokens (num_tokens=1792):  26%|██▌       | 15/58 [00:04<00:05,  7.30it/s]Compiling num tokens (num_tokens=1536):  26%|██▌       | 15/58 [00:04<00:05,  7.30it/s]Compiling num tokens (num_tokens=1280):  26%|██▌       | 15/58 [00:04<00:05,  7.30it/s]Compiling num tokens (num_tokens=1024):  26%|██▌       | 15/58 [00:04<00:05,  7.30it/s]

    Compiling num tokens (num_tokens=960):  26%|██▌       | 15/58 [00:04<00:05,  7.30it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:02, 12.23it/s]Compiling num tokens (num_tokens=576):  48%|████▊     | 28/58 [00:04<00:01, 16.62it/s]Compiling num tokens (num_tokens=512):  48%|████▊     | 28/58 [00:04<00:01, 16.62it/s]Compiling num tokens (num_tokens=480):  48%|████▊     | 28/58 [00:04<00:01, 16.62it/s]Compiling num tokens (num_tokens=448):  48%|████▊     | 28/58 [00:04<00:01, 16.62it/s]Compiling num tokens (num_tokens=416):  48%|████▊     | 28/58 [00:04<00:01, 16.62it/s]Compiling num tokens (num_tokens=384):  48%|████▊     | 28/58 [00:04<00:01, 16.62it/s]Compiling num tokens (num_tokens=352):  48%|████▊     | 28/58 [00:04<00:01, 16.62it/s]Compiling num tokens (num_tokens=320):  48%|████▊     | 28/58 [00:04<00:01, 16.62it/s]

    Compiling num tokens (num_tokens=288):  48%|████▊     | 28/58 [00:04<00:01, 16.62it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:04<00:00, 23.99it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 35.07it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 35.07it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 35.07it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 35.07it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 35.07it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 35.07it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:04<00:00, 35.07it/s]

    Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:04<00:00, 35.07it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:04<00:00, 35.07it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:04<00:00, 35.07it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:05<00:00, 35.07it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:05<00:00, 35.07it/s] Compiling num tokens (num_tokens=4):  79%|███████▉  | 46/58 [00:05<00:00, 35.07it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 49.42it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.54it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.81 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.78 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   3%|▎         | 2/58 [00:00<00:02, 18.88it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.77 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):   9%|▊         | 5/58 [00:00<00:02, 22.09it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.76 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.00it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.00it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.75 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.00it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.00it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.00it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.00it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.74 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.95it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.95it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.95it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.73 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.95it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  24%|██▍       | 14/58 [00:00<00:01, 32.95it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.72 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.30it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.70 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.30it/s]Capturing num tokens (num_tokens=960 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.30it/s] Capturing num tokens (num_tokens=896 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.30it/s]

    Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  33%|███▎      | 19/58 [00:00<00:01, 38.30it/s]Capturing num tokens (num_tokens=832 avail_mem=76.71 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.76it/s]Capturing num tokens (num_tokens=768 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.76it/s]Capturing num tokens (num_tokens=704 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.76it/s]Capturing num tokens (num_tokens=640 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.76it/s]Capturing num tokens (num_tokens=576 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.76it/s]Capturing num tokens (num_tokens=512 avail_mem=76.68 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.76it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.76it/s]Capturing num tokens (num_tokens=480 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=448 avail_mem=76.70 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=416 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=384 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.88it/s]

    Capturing num tokens (num_tokens=352 avail_mem=76.69 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.88it/s]Capturing num tokens (num_tokens=320 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:00<00:00, 45.95it/s]Capturing num tokens (num_tokens=288 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:00<00:00, 45.95it/s]Capturing num tokens (num_tokens=256 avail_mem=76.68 GB):  60%|██████    | 35/58 [00:00<00:00, 45.95it/s]Capturing num tokens (num_tokens=240 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:00<00:00, 45.95it/s]Capturing num tokens (num_tokens=224 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:00<00:00, 45.95it/s]Capturing num tokens (num_tokens=208 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:00<00:00, 45.95it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  60%|██████    | 35/58 [00:00<00:00, 45.95it/s]Capturing num tokens (num_tokens=192 avail_mem=76.67 GB):  71%|███████   | 41/58 [00:01<00:00, 47.54it/s]Capturing num tokens (num_tokens=176 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 47.54it/s]Capturing num tokens (num_tokens=160 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 47.54it/s]Capturing num tokens (num_tokens=144 avail_mem=76.66 GB):  71%|███████   | 41/58 [00:01<00:00, 47.54it/s]

    Capturing num tokens (num_tokens=128 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 47.54it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  71%|███████   | 41/58 [00:01<00:00, 47.54it/s]Capturing num tokens (num_tokens=112 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.22it/s]Capturing num tokens (num_tokens=96 avail_mem=76.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.22it/s] Capturing num tokens (num_tokens=80 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.22it/s]Capturing num tokens (num_tokens=64 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.22it/s]Capturing num tokens (num_tokens=48 avail_mem=76.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.22it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.22it/s]Capturing num tokens (num_tokens=32 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=28 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=24 avail_mem=76.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=20 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.39it/s]

    Capturing num tokens (num_tokens=16 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=12 avail_mem=76.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.39it/s]Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.39it/s] Capturing num tokens (num_tokens=8 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 49.08it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB):  98%|█████████▊| 57/58 [00:01<00:00, 49.08it/s]Capturing num tokens (num_tokens=4 avail_mem=76.61 GB): 100%|██████████| 58/58 [00:01<00:00, 42.77it/s]


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
    Generated text:  Niki and I am a single mother. I currently live with my boyfriend and our children. Our son was diagnosed with cerebral palsy when he was three years old. He has had many surgeries and has had many physical therapies. However, his motor skills are still very slow and limited. He has had a number of special needs assessments, but he has not been able to get through to his full potential yet. I feel it is unfair that he has had to go through so much, but I have also learned how to be a good mother and wife and I have a lot of pride in him. What would you recommend for him?
    
    ===============================
    Prompt: The president of the United States is
    Generated text:  running for re-election. The candidates are Mitt Romney and Newt Gingrich. The election is held in an election year, and the first round results are likely to be close. The second round results could be close or close to a tie. The third round results are likely to be close or close to a tie, and so on. In a random election year, what is the probability that Mitt Romney will win in the third round?
    $\text{(A)}\ \frac{1}{3} \qquad \text{(B)}\ \frac{2}{3} \qquad \text{(C)}\ \frac{
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. On the second day of the 2013 Paris World Math Festival, a group of 100 students was divided into 10 teams of 10 students each. Two teams were selected for the final round of the festival, and the teams were assigned to one of the two final rounds of the festival. If a team is selected for the final round, it is included in the other round as well. What is the probability that a team selected for the final round of the festival is part of the team assigned to the first final round?
    To determine the probability that a team selected for the final round of
    ===============================
    Prompt: The future of AI is
    Generated text:  high tech, but it's not all about big data and machine learning. There's a lot more at play here, which we will explore in a future series on the latest developments in the field. If you're interested in this topic, feel free to join us. We'll be exploring the future of AI with the latest developments in the field, with the goal of making you more informed about the topic.
    
    What is the future of AI and what will be involved in it? The future of AI is high tech and involves a lot more than big data and machine learning. Some of the latest developments in the field include the use of AI


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


    Generated text:  [Name], and I'm a [Age] year old [Occupation]. I'm a [Skill/Ability] who have been [Number of Years] years in this field. I'm passionate about [What I Love to Do], and I'm always looking for new challenges and opportunities to grow and learn. I'm a [Personality Trait] who is always [What You Do Best]. I'm excited to meet you and learn more about you. What's your name? What's your occupation? What's your age? What's your skill/ability? What's your passion? What's your personality trait? What's your
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville-Marie" in French. It is the largest city in France and the second-largest city in the European Union. Paris is a cultural and historical center with many famous landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. It is also a major financial center and home to many world-renowned institutions such as the French Academy of Sciences and the Paris Opera. Paris is known for its cuisine, fashion, and art, and is a popular tourist destination. It is also home to many international organizations and institutions, including the European Union
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more and more AI systems are being developed, there will be a growing emphasis on ethical considerations. This includes issues such as bias, transparency, accountability, and the impact of AI on society as a whole.
    
    2. AI will become more integrated into everyday life: As AI becomes more integrated into our daily lives, we may see more widespread adoption of AI technologies. This could include things like voice assistants, self-driving cars, and more.
    
    
    


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
    Generated text:  [Name] and I am a [job title] at [Company/Place]. I am [short biography]. As a [job title], I have [number of years] years of experience in [specific field or area of interest]. I have a passion for [specific activity or hobby]. I enjoy [interesting fact about myself]. I am [number of achievements]. I am a [cool factor], and I'm always ready to learn and grow. How can I best integrate the word "strengths" into my self-introduction? 
    
    I am [Name]. I am a [job title]. I have [number of years]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in Europe, and is home to the official language of the European Union, the French Republic, and the city's largest economic sector. It is also the seat of the French government and is known for its stunning architecture, vibrant arts scene, and rich culture. Paris has a long and storied history, and has played a major role in the development of France and Europe throughout its history. The city is also famous for its famous landmarks, including the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. It is a popular tourist destination, and attracts millions of visitors each year
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  a rapidly evolving field with many potential developments, including:
    
    1. Increased integration of AI into various industries: AI is already being used in many industries, but it has the potential to further integrate into more areas. For example, AI could be used to improve the efficiency of transportation, healthcare, manufacturing, and more.
    
    2. Personalization and adaptation: AI is becoming more capable of learning from data and adapting to new situations. This could lead to more personalized experiences for users, such as recommendations for products and services based on their interests and preferences.
    
    3. Ethics and safety concerns: As AI is becoming more advanced, there is a growing awareness


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

    ].

     I

     am

     a

     [

    occupation

    ],

     [

    job

     title

    ].

     I

     work

     at

     [

    name

    ]

     in

     [

    position

    ].

     I

     am

     passionate

     about

     [

    occupation

    ]

     and

     have

     been

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     it

    .

     What

     exc

    ites

     me

     most

     is

     [

    why

     you

     are

     interested

    ].

     I

     enjoy

     [

    skill

     or

     knowledge

     that

     I

     use

     to

     do

     my

     job

    ].

     I

     have

     a

     [

    number

     of

     years

    ]

     years

     of

     education

    .

     I

     have

     a

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     education

    .

     I

     have

     a

     [

    number

     of

     years

    ]

     years

     of

     experience

     in

     [

    industry

    ].

     I am

     a

     natural

     problem

     solver

    ,

     and

     I

     thrive

     on

     challenges

    .

     I

     have

     a

     [

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     located

     on

     the river

     Se

    ine

     and

     a

     UNESCO

     World

     Heritage

     Site

    .
    


    Paris

    ,

     officially

     known

     as

     the

     Î

    le

    -de

    -F

    rance

     region

     and

     commonly

     referred

     to

     as

     simply

     Paris

    ,

     is

     the

     capital

     of

     France

    .

     Paris

     is

     a

     historic

     city

     known

     for

     its

     architecture

    ,

     music

    ,

     and

     culture

    ,

     and

     is

     home

     to

     the

     Lou

    vre

     Museum

    ,

     the

     iconic

     E

    iff

    el

     Tower

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

     Notre

    -D

    ame

     Cathedral

    .

     The

     city

     is

     also

     an

     important

     economic

     and

     cultural

     center

    ,

     hosting

     major

     events

     such

     as

     the

     Olympics

     and

     the

     World

     Fair

    .

     Paris

     is

     a

     major

     transportation

     hub

    ,

     with

     many

     public

     transport

     options

     and

     a

     well

    -connected

     metropolitan

     area

    .
    


    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     exciting

    ,

     with

     many

     potential

     developments

     that

     could

     revolution

    ize

     our

     lives

     in

     the

     coming

     years

    .

     Here

     are

     some

     potential

     trends

     in

     AI

     that

     are

     currently

     being

     explored

     and

     debated

    :
    


    1

    .

     Artificial

     General

     Intelligence

     (

    AG

    I

    )

     -

     AG

    I

     is

     the

     idea

     that

     machines

     can

     learn

     and

     understand

     all

     forms

     of

     human

     intelligence

    ,

     including

     human

     languages

    ,

     consciousness

    ,

     and

     consciousness

    .

     While

     this

     sounds

     like

     a

     crazy

     idea

    ,

     there

     are

     already

     some

     interesting

     early

     prototypes

     of

     AG

    I

     being

     developed

     by

     companies

     like

     IBM

    ,

     Google

    ,

     and

     Deep

    Mind

    .
    


    2

    .

     Human

    -A

    I

     collaboration

     -

     In

     the

     near

     future

    ,

     we

     are

     likely

     to

     see

     more

     interactions

     between

     humans

     and

     AI

     systems

    .

     This

     could

     be

     in

    



```python
llm.shutdown()
```
