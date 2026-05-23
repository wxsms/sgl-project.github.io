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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.91it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  3.91it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:41,  3.88s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:41,  3.88s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:41,  3.88s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:41,  3.88s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:03<03:41,  3.88s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:03<00:31,  1.67it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:03<00:31,  1.67it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:31,  1.67it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.92it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.92it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.92it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.92it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.92it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.92it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.92it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.92it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.92it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.92it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:04<00:03, 10.31it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:04<00:03, 10.31it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:04<00:03, 10.31it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:04<00:03, 10.31it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:04<00:03, 10.31it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:04<00:03, 10.31it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:04<00:03, 10.31it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:04<00:03, 10.31it/s]

    Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:04<00:03, 10.31it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:04<00:01, 16.07it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:04<00:01, 21.77it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:04<00:01, 21.77it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:04<00:01, 21.77it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:04<00:01, 21.77it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:04<00:01, 21.77it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:04<00:01, 21.77it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:04<00:01, 21.77it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:04<00:01, 21.77it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:04<00:01, 21.77it/s]Compiling num tokens (num_tokens=128):  62%|██████▏   | 36/58 [00:04<00:01, 21.77it/s]

    Compiling num tokens (num_tokens=112):  62%|██████▏   | 36/58 [00:04<00:01, 21.77it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=28):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=24):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=20):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=16):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=12):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=8):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s] Compiling num tokens (num_tokens=4):  79%|███████▉  | 46/58 [00:04<00:00, 31.98it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 45.45it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.55it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.24it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   3%|▎         | 2/58 [00:00<00:02, 19.24it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=74.17 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.16 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):   9%|▊         | 5/58 [00:00<00:02, 22.40it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.29it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.15 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.29it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.29it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.14 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.29it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.29it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.29it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.13 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=1792 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  24%|██▍       | 14/58 [00:00<00:01, 33.08it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.12 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.89it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.89it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.09 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.89it/s]Capturing num tokens (num_tokens=960 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.89it/s] Capturing num tokens (num_tokens=896 avail_mem=74.11 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.89it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  33%|███▎      | 19/58 [00:00<00:01, 37.89it/s]Capturing num tokens (num_tokens=832 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=768 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=704 avail_mem=74.10 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=640 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=576 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=512 avail_mem=74.08 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  41%|████▏     | 24/58 [00:00<00:00, 41.63it/s]Capturing num tokens (num_tokens=480 avail_mem=74.09 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=448 avail_mem=74.09 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=416 avail_mem=74.09 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=384 avail_mem=74.09 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.71it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.08 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  52%|█████▏    | 30/58 [00:00<00:00, 44.71it/s]Capturing num tokens (num_tokens=320 avail_mem=74.08 GB):  60%|██████    | 35/58 [00:00<00:00, 45.91it/s]Capturing num tokens (num_tokens=288 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:00<00:00, 45.91it/s]Capturing num tokens (num_tokens=256 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:00<00:00, 45.91it/s]Capturing num tokens (num_tokens=240 avail_mem=74.07 GB):  60%|██████    | 35/58 [00:00<00:00, 45.91it/s]Capturing num tokens (num_tokens=224 avail_mem=74.06 GB):  60%|██████    | 35/58 [00:00<00:00, 45.91it/s]Capturing num tokens (num_tokens=208 avail_mem=74.06 GB):  60%|██████    | 35/58 [00:00<00:00, 45.91it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  60%|██████    | 35/58 [00:00<00:00, 45.91it/s]Capturing num tokens (num_tokens=192 avail_mem=74.06 GB):  71%|███████   | 41/58 [00:01<00:00, 47.51it/s]Capturing num tokens (num_tokens=176 avail_mem=74.06 GB):  71%|███████   | 41/58 [00:01<00:00, 47.51it/s]Capturing num tokens (num_tokens=160 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 47.51it/s]Capturing num tokens (num_tokens=144 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 47.51it/s]

    Capturing num tokens (num_tokens=128 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 47.51it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  71%|███████   | 41/58 [00:01<00:00, 47.51it/s]Capturing num tokens (num_tokens=112 avail_mem=74.05 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.16it/s]Capturing num tokens (num_tokens=96 avail_mem=74.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.16it/s] Capturing num tokens (num_tokens=80 avail_mem=74.04 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.16it/s]Capturing num tokens (num_tokens=64 avail_mem=74.03 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.16it/s]Capturing num tokens (num_tokens=48 avail_mem=74.03 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.16it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  79%|███████▉  | 46/58 [00:01<00:00, 48.16it/s]Capturing num tokens (num_tokens=32 avail_mem=74.03 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.08it/s]Capturing num tokens (num_tokens=28 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.08it/s]Capturing num tokens (num_tokens=24 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.08it/s]Capturing num tokens (num_tokens=20 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.08it/s]

    Capturing num tokens (num_tokens=16 avail_mem=74.02 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.08it/s]Capturing num tokens (num_tokens=12 avail_mem=74.01 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.08it/s]Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  88%|████████▊ | 51/58 [00:01<00:00, 48.08it/s] Capturing num tokens (num_tokens=8 avail_mem=74.01 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.89it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB):  98%|█████████▊| 57/58 [00:01<00:00, 48.89it/s]Capturing num tokens (num_tokens=4 avail_mem=74.00 GB): 100%|██████████| 58/58 [00:01<00:00, 42.73it/s]


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
    Generated text:  Bob. I have been a member of the club for about a year. I like to play soccer. I play soccer with my friends every Saturday. We play soccer at the playground. We play on the grass. We have lots of fun playing soccer. I really like playing soccer. I often play with my friends after school. I like playing with my friends. We have fun playing soccer. We often play with my brother. He likes playing soccer too. We have fun playing soccer. We play soccer at the playground on the grass. We have lots of fun playing soccer. What can we learn about Bob?
    A) He is a
    ===============================
    Prompt: The president of the United States is
    Generated text:  a political leader, and the U. S. Congress is the legislative branch of the federal government. The U. S. Congress consists of two chambers, the House of Representatives and the Senate. The president is elected by the electorate and has no political power, while the House of Representatives and the Senate are elected through the processes of voting in elections. Both chambers of Congress are accountable to the people for their actions.
    Does this next sentence follow, given the above text?
    The U. S. Congress is the legislative branch of the federal government.
    Choices:
    a). yes
    b). it is not possible to tell
    c). no
    
    
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris.
    A. True
    B. False
    Answer:
    
    A
    
    A. Correct
    B. Incorrect
    Answer:
    
    A
    
    A. Correct
    B. Incorrect
    Answer:
    
    B
    
    A. Correct
    B. Incorrect
    Answer:
    
    B
    
    A. Correct
    B. Incorrect
    Answer:
    
    A
    
    A. Correct
    B. Incorrect
    Answer:
    
    B
    
    A. Correct
    B. Incorrect
    Answer:
    
    A
    
    A. Correct
    B. Incorrect
    Answer:
    
    A
    
    A. Correct
    B. Incorrect
    Answer:
    
    B
    
    A. Correct
    B. Incorrect
    Answer:
    
    B
    
    A. Correct
    B.
    ===============================
    Prompt: The future of AI is
    Generated text:  already here. The technology is already developing and growing rapidly, and it has the potential to revolutionize the world and make our lives better. AI is already making a huge impact on our world, from smart homes and virtual assistants to personalized marketing and healthcare. As technology continues to advance, it's likely that we will see even more incredible applications of AI in the years to come.
    How can AI be made to be more accessible and affordable for everyone? Here are some ideas:
    
    1. Implementing AI in schools: AI can be used in classrooms to help students learn. For example, AI can be used to create personalized learning paths for students


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


    Generated text:  Paris, also known as the City of Light. It is a historic city with a rich cultural heritage and is home to many famous landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also known for its vibrant nightlife, fashion industry, and annual festivals such as the Eiffel Tower Parade and the Carnaval de Paris. The city is also home to many international organizations and institutions, including the French Academy of Sciences and the French National Library. Paris is a major cultural and economic center in Europe and is a popular tourist destination. The city is also known for its diverse cuisine, including
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the way we live, work, and interact with technology. Here are some of the most likely future trends in AI:
    
    1. Increased automation and robotics: As AI continues to advance, we can expect to see more automation and robotics in various industries, from manufacturing to healthcare. This will likely lead to increased efficiency and productivity, but it will also create new jobs and challenges for workers.
    
    2. AI-powered healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As AI technology continues to advance, we can expect to see even more innovative applications
    


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
    Generated text:  [Name], and I'm a [role] with over [number] years of experience in [occupation]. I'm a... [insert personality or professional trait here]. If you wanted to meet me, I would say, "Hello! I'm [Name], a [role] with over [number] years of experience in [occupation]. I'm a [insert personality or professional trait here]. If you wanted to meet me, I would say, "Hello! I'm [Name], a [role] with over [number] years of experience in [occupation]. I'm a [insert personality or professional trait here]. " [
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a city located in the heart of the French countryside. It is the largest city in France by population, with an estimated population of over 1. 3 million people. Paris is known for its historical architecture, vibrant culture, and beautiful views of the French countryside. It is also home to many world-renowned museums, art galleries, and opera houses, making it a major center of culture and entertainment in the country. Paris is also home to many significant religious and political institutions, including the Eiffel Tower and the Louvre Museum. As the capital of France, Paris plays a crucial role in shaping the country's identity
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  rapidly changing, and it is expected to continue to evolve in exciting ways. Here are some possible trends in AI that could impact the industry in the coming years:
    
    1. More autonomous and self-driving vehicles: Autonomous vehicles have been on the horizon for years, and we are rapidly moving towards a world where these vehicles are the norm. The development of more advanced AI systems for self-driving cars and trucks is expected to drive the growth of the autonomous vehicle industry.
    
    2. Increased focus on ethical AI: As AI systems become more complex, there is growing concern about their potential to cause harm or be biased. This is driving a focus on ethical AI


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

    ____________

    _.

     I

     am

     a

     __

    ____________

    _.

     I

     love

     __

    ________

    _

     because

     __

    ________

    __.

     What

     are

     you

     doing

     right

     now

    ?

     What

     are

     you

     looking

     forward

     to

    ?

     What

     do

     you

     like

     to

     do

    ?

     What

     are

     you

     into

    ?

     What

     are

     you

     into

    ?

     What

     is

     your

     interests

    ?

     I

     can

    't

     help

     but

     laugh

     at

     __

    ________

    ___

    '

    s

     jokes

    ,

     but

     I

     find

     them

     amusing

    .

     I

     have

     a

     ___

     favourite

     sport

    ,

     which

     I

     enjoy

     watching

     and

     playing

    .

     I

     like

     to

     eat

     __

    ________

    _

     and

     __

    ________

    ___

     and

     like

     to

     go

     for

     __

    ________

    _

     and

     __

    ________

    _

     on

     weekends

    .

     I

     enjoy

     ____

    _.

     I

     like

     to

     write

     __

    ________

    _,

     __

    ________

    _,

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     iconic

     architecture

    ,

     rich

     history

    ,

     and

     vibrant

     culture

    .

     Paris

     was

     founded

     in

     the

     

    1

    2

    th

     century

     as

     the

     capital

     of

     the

     Duch

    y

     of

     Paris

     and

     became

     the

     capital

     of

     France

     in

     

    1

    8

    0

    0

    ,

     under

     Napoleon

     Bon

    ap

    arte

    .

     The

     city

     is

     home

     to

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

     Arc

     de

     Tri

    omp

    he

    ,

     among

     others

    .

     Paris

     is

     also

     known

     for

     its

     food

     and

     drinks

     scene

    ,

     with

     its

     famous

     bist

    ros

    ,

     cafes

    ,

     and

     restaurants

     serving

     up

     delicious

     cuisine

     and

     drinks

    .

     Overall

    ,

     Paris

     is

     a

     city

     of

     art

    ,

     culture

    ,

     and

     history

     that

     is

     widely

     regarded

     as

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     highly

     unpredictable

     and

     difficult

     to

     predict

     with

     certainty

    .

     However

    ,

     some

     possible

     trends

     that

     have

     been

     observed

     in

     the

     field

     include

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     in

     healthcare

    :

     As

     AI

     becomes

     more

     advanced

    ,

     it

     will

     be

     used

     to

     improve

     patient

     care

    ,

     reduce

     errors

    ,

     and

     increase

     efficiency

     in

     healthcare

    .
    


    2

    .

     Greater

     automation

     in

     manufacturing

    :

     With

     AI

    ,

     manufacturing

     processes

     can

     be

     automated

    ,

     reducing

     the

     need

     for

     human

     labor

     and

     increasing

     production

     efficiency

    .
    


    3

    .

     AI

     integration

     in

     customer

     service

    :

     AI

     can

     be

     used

     to

     automate

     customer

     service

     interactions

    ,

     improving

     response

     times

     and

     improving

     customer

     satisfaction

    .
    


    4

    .

     Enhanced

     autonomous

     vehicles

    :

     Autonomous

     vehicles

     could

     become

     more

     prevalent

     in

     the

     future

    ,

     leading

     to

     a

    



```python
llm.shutdown()
```
