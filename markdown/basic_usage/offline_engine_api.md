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

    Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.58it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  4.58it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:23,  5.68s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:23,  5.68s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:05<05:23,  5.68s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:05<05:23,  5.68s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:59,  1.11s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:59,  1.11s/it]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:05<00:59,  1.11s/it]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:05<00:59,  1.11s/it]

    Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:05<00:59,  1.11s/it]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:22,  2.22it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:22,  2.22it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:22,  2.22it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:22,  2.22it/s]Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:22,  2.22it/s]Compiling num tokens (num_tokens=3072):  14%|█▍        | 8/58 [00:05<00:22,  2.22it/s]Compiling num tokens (num_tokens=2816):  14%|█▍        | 8/58 [00:05<00:22,  2.22it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:06<00:09,  4.86it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:06<00:09,  4.86it/s]Compiling num tokens (num_tokens=2304):  24%|██▍       | 14/58 [00:06<00:09,  4.86it/s]Compiling num tokens (num_tokens=2048):  24%|██▍       | 14/58 [00:06<00:09,  4.86it/s]Compiling num tokens (num_tokens=1792):  24%|██▍       | 14/58 [00:06<00:09,  4.86it/s]Compiling num tokens (num_tokens=1536):  24%|██▍       | 14/58 [00:06<00:09,  4.86it/s]Compiling num tokens (num_tokens=1280):  24%|██▍       | 14/58 [00:06<00:09,  4.86it/s]

    Compiling num tokens (num_tokens=1024):  24%|██▍       | 14/58 [00:06<00:09,  4.86it/s]Compiling num tokens (num_tokens=1024):  36%|███▌      | 21/58 [00:06<00:04,  8.85it/s]Compiling num tokens (num_tokens=960):  36%|███▌      | 21/58 [00:06<00:04,  8.85it/s] Compiling num tokens (num_tokens=896):  36%|███▌      | 21/58 [00:06<00:04,  8.85it/s]Compiling num tokens (num_tokens=832):  36%|███▌      | 21/58 [00:06<00:04,  8.85it/s]Compiling num tokens (num_tokens=768):  36%|███▌      | 21/58 [00:06<00:04,  8.85it/s]Compiling num tokens (num_tokens=704):  36%|███▌      | 21/58 [00:06<00:04,  8.85it/s]Compiling num tokens (num_tokens=640):  36%|███▌      | 21/58 [00:06<00:04,  8.85it/s]Compiling num tokens (num_tokens=576):  36%|███▌      | 21/58 [00:06<00:04,  8.85it/s]Compiling num tokens (num_tokens=512):  36%|███▌      | 21/58 [00:06<00:04,  8.85it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:06<00:02, 14.43it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:06<00:02, 14.43it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:06<00:02, 14.43it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:06<00:02, 14.43it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:06<00:02, 14.43it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:06<00:02, 14.43it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:06<00:02, 14.43it/s]

    Compiling num tokens (num_tokens=288):  50%|█████     | 29/58 [00:06<00:02, 14.43it/s]Compiling num tokens (num_tokens=256):  50%|█████     | 29/58 [00:06<00:02, 14.43it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:06<00:01, 20.78it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:06<00:00, 26.68it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:06<00:00, 26.68it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:06<00:00, 26.68it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:06<00:00, 26.68it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:06<00:00, 26.68it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:06<00:00, 26.68it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:06<00:00, 26.68it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:06<00:00, 26.68it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:06<00:00, 33.17it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:06<00:00, 33.17it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:06<00:00, 33.17it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:06<00:00, 33.17it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:06<00:00, 33.17it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:06<00:00, 33.17it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:06<00:00, 33.17it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:06<00:00, 33.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:06<00:00,  8.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=72.59 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.56 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=72.56 GB):   3%|▎         | 2/58 [00:00<00:03, 15.83it/s]Capturing num tokens (num_tokens=7168 avail_mem=72.56 GB):   3%|▎         | 2/58 [00:00<00:03, 15.83it/s]Capturing num tokens (num_tokens=6656 avail_mem=72.56 GB):   3%|▎         | 2/58 [00:00<00:03, 15.83it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=72.56 GB):   3%|▎         | 2/58 [00:00<00:03, 15.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=72.56 GB):   9%|▊         | 5/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=5632 avail_mem=72.55 GB):   9%|▊         | 5/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=5120 avail_mem=72.54 GB):   9%|▊         | 5/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.11 GB):   9%|▊         | 5/58 [00:00<00:02, 18.18it/s]Capturing num tokens (num_tokens=4608 avail_mem=72.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=4096 avail_mem=72.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.96it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=72.11 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.10 GB):  14%|█▍        | 8/58 [00:00<00:02, 20.96it/s]Capturing num tokens (num_tokens=3584 avail_mem=72.10 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=3328 avail_mem=72.10 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=3072 avail_mem=72.09 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.09 GB):  19%|█▉        | 11/58 [00:00<00:02, 21.95it/s]Capturing num tokens (num_tokens=2816 avail_mem=72.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.55it/s]Capturing num tokens (num_tokens=2560 avail_mem=72.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.55it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=72.09 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.55it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.08 GB):  24%|██▍       | 14/58 [00:00<00:01, 23.55it/s]Capturing num tokens (num_tokens=2048 avail_mem=72.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.94it/s]Capturing num tokens (num_tokens=1792 avail_mem=72.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.94it/s]Capturing num tokens (num_tokens=1536 avail_mem=72.08 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.94it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.07 GB):  29%|██▉       | 17/58 [00:00<00:01, 23.94it/s]Capturing num tokens (num_tokens=1280 avail_mem=72.07 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.32it/s]Capturing num tokens (num_tokens=1024 avail_mem=72.06 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.32it/s]

    Capturing num tokens (num_tokens=960 avail_mem=72.07 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.32it/s] Capturing num tokens (num_tokens=896 avail_mem=72.07 GB):  34%|███▍      | 20/58 [00:00<00:01, 25.32it/s]Capturing num tokens (num_tokens=896 avail_mem=72.07 GB):  40%|███▉      | 23/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=832 avail_mem=72.06 GB):  40%|███▉      | 23/58 [00:00<00:01, 25.36it/s]Capturing num tokens (num_tokens=768 avail_mem=72.06 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.36it/s]Capturing num tokens (num_tokens=704 avail_mem=72.06 GB):  40%|███▉      | 23/58 [00:01<00:01, 25.36it/s]Capturing num tokens (num_tokens=704 avail_mem=72.06 GB):  45%|████▍     | 26/58 [00:01<00:01, 25.34it/s]Capturing num tokens (num_tokens=640 avail_mem=72.05 GB):  45%|████▍     | 26/58 [00:01<00:01, 25.34it/s]

    Capturing num tokens (num_tokens=576 avail_mem=72.05 GB):  45%|████▍     | 26/58 [00:01<00:01, 25.34it/s]Capturing num tokens (num_tokens=512 avail_mem=72.04 GB):  45%|████▍     | 26/58 [00:01<00:01, 25.34it/s]Capturing num tokens (num_tokens=512 avail_mem=72.04 GB):  50%|█████     | 29/58 [00:01<00:01, 24.61it/s]Capturing num tokens (num_tokens=480 avail_mem=72.05 GB):  50%|█████     | 29/58 [00:01<00:01, 24.61it/s]Capturing num tokens (num_tokens=448 avail_mem=72.05 GB):  50%|█████     | 29/58 [00:01<00:01, 24.61it/s]Capturing num tokens (num_tokens=416 avail_mem=72.05 GB):  50%|█████     | 29/58 [00:01<00:01, 24.61it/s]

    Capturing num tokens (num_tokens=416 avail_mem=72.05 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.30it/s]Capturing num tokens (num_tokens=384 avail_mem=72.05 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.30it/s]Capturing num tokens (num_tokens=352 avail_mem=72.04 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.30it/s]Capturing num tokens (num_tokens=320 avail_mem=72.04 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.30it/s]Capturing num tokens (num_tokens=288 avail_mem=72.03 GB):  55%|█████▌    | 32/58 [00:01<00:01, 25.30it/s]Capturing num tokens (num_tokens=288 avail_mem=72.03 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.30it/s]Capturing num tokens (num_tokens=256 avail_mem=72.03 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.30it/s]Capturing num tokens (num_tokens=240 avail_mem=72.03 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.30it/s]

    Capturing num tokens (num_tokens=224 avail_mem=72.02 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.30it/s]Capturing num tokens (num_tokens=208 avail_mem=72.02 GB):  62%|██████▏   | 36/58 [00:01<00:00, 26.30it/s]Capturing num tokens (num_tokens=208 avail_mem=72.02 GB):  69%|██████▉   | 40/58 [00:01<00:00, 27.81it/s]Capturing num tokens (num_tokens=192 avail_mem=72.02 GB):  69%|██████▉   | 40/58 [00:01<00:00, 27.81it/s]Capturing num tokens (num_tokens=176 avail_mem=72.02 GB):  69%|██████▉   | 40/58 [00:01<00:00, 27.81it/s]Capturing num tokens (num_tokens=160 avail_mem=72.01 GB):  69%|██████▉   | 40/58 [00:01<00:00, 27.81it/s]Capturing num tokens (num_tokens=144 avail_mem=72.01 GB):  69%|██████▉   | 40/58 [00:01<00:00, 27.81it/s]Capturing num tokens (num_tokens=144 avail_mem=72.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.13it/s]Capturing num tokens (num_tokens=128 avail_mem=72.01 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.13it/s]

    Capturing num tokens (num_tokens=112 avail_mem=72.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.13it/s]Capturing num tokens (num_tokens=96 avail_mem=72.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.13it/s] Capturing num tokens (num_tokens=80 avail_mem=72.00 GB):  76%|███████▌  | 44/58 [00:01<00:00, 29.13it/s]Capturing num tokens (num_tokens=80 avail_mem=72.00 GB):  83%|████████▎ | 48/58 [00:01<00:00, 29.90it/s]Capturing num tokens (num_tokens=64 avail_mem=71.99 GB):  83%|████████▎ | 48/58 [00:01<00:00, 29.90it/s]Capturing num tokens (num_tokens=48 avail_mem=71.99 GB):  83%|████████▎ | 48/58 [00:01<00:00, 29.90it/s]Capturing num tokens (num_tokens=32 avail_mem=71.99 GB):  83%|████████▎ | 48/58 [00:01<00:00, 29.90it/s]Capturing num tokens (num_tokens=28 avail_mem=71.98 GB):  83%|████████▎ | 48/58 [00:01<00:00, 29.90it/s]

    Capturing num tokens (num_tokens=28 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 30.49it/s]Capturing num tokens (num_tokens=24 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:01<00:00, 30.49it/s]Capturing num tokens (num_tokens=20 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:02<00:00, 30.49it/s]Capturing num tokens (num_tokens=16 avail_mem=71.98 GB):  90%|████████▉ | 52/58 [00:02<00:00, 30.49it/s]Capturing num tokens (num_tokens=12 avail_mem=71.97 GB):  90%|████████▉ | 52/58 [00:02<00:00, 30.49it/s]Capturing num tokens (num_tokens=12 avail_mem=71.97 GB):  97%|█████████▋| 56/58 [00:02<00:00, 31.10it/s]Capturing num tokens (num_tokens=8 avail_mem=71.97 GB):  97%|█████████▋| 56/58 [00:02<00:00, 31.10it/s] Capturing num tokens (num_tokens=4 avail_mem=71.96 GB):  97%|█████████▋| 56/58 [00:02<00:00, 31.10it/s]Capturing num tokens (num_tokens=4 avail_mem=71.96 GB): 100%|██████████| 58/58 [00:02<00:00, 26.80it/s]


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
    Generated text:  Martin Bell and I am a digital artist based in Wellington, New Zealand. My work is based on the concept of a "time signature" and how it can be used as a tool to explore different forms of abstract art. My artwork is inspired by a personal journey, and I use the concept of "a time signature" as a visual metaphor to explore how time is being reinterpreted in different ways and how we represent it in our everyday lives.
    I create digital art and marketing design, especially for events, weddings, and photography, and I focus on the use of the "time signature" to create a new, unique visual language in
    ===============================
    Prompt: The president of the United States is
    Generated text:  a position that is given to someone by the country's government. This position is called the president of the United States. No one in the United States can hold that position. There is no other person in the United States who can hold that position. The president is the chief executive of the United States. That is, the president is in charge of the country's government. The president can appoint other people to help him. He can also cancel the appointments that other people have made to help him. The president can also remove other people from the government. He can do this if he thinks that the other people did something wrong. The president
    ===============================
    Prompt: The capital of France is
    Generated text:  ______.
    A. Paris
    B. Rome
    C. London
    D. Berlin
    Answer:
    
    A
    
    The Palace Museum is located in the ______ of Beijing.
    A. City center
    B. Old City
    C. Old City Wall
    D. Forbidden City
    Answer:
    
    C
    
    In the past decade, the number of forest fire incidents in various regions of China has decreased, and the country's fire management has improved. However, the proportion of fire-fighting personnel is still very low. The main reason for this is that the country has not put in place more effective fire-fighting measures. Therefore, the area of forest fire
    ===============================
    Prompt: The future of AI is
    Generated text:  on the horizon. In an era of rapid technological advancement, it is important to be prepared for the impact of AI on the global economy and society. One of the most important areas where AI will have a significant impact is in the field of healthcare. AI can help healthcare providers deliver more efficient, accurate, and personalized care to patients.
    
    AI is already being used in healthcare to help diagnose and treat diseases. For example, AI algorithms can analyze medical imaging data to identify patterns and abnormalities that may be missed by human specialists. This can help early detect diseases such as cancer and heart disease, allowing for faster treatment and improved patient outcomes.
    
    AI is


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


    Generated text:  [Name] and I am a [job title] at [company name]. I have been working in this field for [number] years and have always been passionate about [job title] and its impact on society. I am always looking for ways to [job title] and make a difference in the world. I am a [job title] who is always looking for new challenges and opportunities to grow and learn. I am a [job title] who is always willing to take on new challenges and learn from my mistakes. I am a [job title] who is always looking for ways to [job title] and make a positive
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its fashion industry, with Paris Fashion Week being one of the largest in the world. The French capital is a vibrant and diverse city with a diverse population and a rich history. It is a city that is constantly evolving and is a major hub for business, education, and entertainment
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. Some possible future trends in AI include:
    
    1. Increased focus on ethical considerations: As AI becomes more integrated into our daily lives, there will be a growing emphasis on ethical considerations. This could lead to more stringent regulations and guidelines for AI development and deployment.
    
    2. Greater integration with human intelligence: AI is already being used to augment human intelligence, such as through chatbots and virtual assistants. As AI becomes more advanced, it may be able to learn and adapt to human behavior, potentially leading to even more sophisticated forms of human intelligence
    


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
    Generated text:  [Your Name]. I'm a [Industry/Field] expert in [Your Expertise]. I have been in this field for [Number of Years] years and have honed my skills through my [Number of Projects] projects. I've worked with [Number of Clients] clients across [Number of Locations] locations. I'm passionate about [Industry/Field], and I'm constantly seeking to learn and improve. Please feel free to ask me any questions you have about [Industry/Field] or anything else. I'm [Number of Years] years of age, and I wear [Number of Hat/Masks/Workwear]
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.
    
    That's correct! The capital city of France is Paris, and it's known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Is there anything else you'd like to know about Paris?
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by four key trends:
    
    1. AI will become more integrated with human intelligence: AI will become more integrated with human intelligence, allowing machines to better understand and respond to the nuances of human emotions, thought processes, and decision-making.
    
    2. AI will become more versatile and adaptable: AI will become more versatile and adaptable, able to learn and improve its performance based on new data and feedback from users.
    
    3. AI will become more human-like: AI will become more human-like, with machines that have the ability to understand, empathize with, and even feel emotions similar to human beings.
    
    4. AI will become more


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

     Name

    ],

     I

    'm

     a

     [

    Age

    ]

     year

     old

     [

    Gender

    ]

     [

    Occup

    ation

    ],

     [

    Character

     Name

    ]

     is

     a

     [

    Type

     of

     Pet

    ]

     owned

     by

     [

    Pet

     Owner

    's

     Name

    ]

     and

     is

     an

     [

    Activity

    ]

     for

     [

    Number

     of

     Months

    ]

     months

    ,

     [

    Character

     Name

    ]

     has

     a

     [

    Person

    ality

     Trait

     or

     Quality

    ]

     and

     [

    Favorite

     Activity

    ]

     that

     makes

     their

     life

     rewarding

     and

     unique

    .

     [

    Character

     Name

    ]

     is

     passionate

     about

     [

    Occup

    ation

    /

    Activity

    ]

     and

     [

    Character

     Name

    ]

     believes

     in

     [

    Dream

    /

    Goal

    ]

     because

     [

    Reason

     for

     Bel

    ieving

    ].

     I

    'm

     [

    Appearance

    /

    Physical

     Characteristics

    ]

     and

     [

    Character

     Name

    ]

     is

     [

    Personal

    ity

     Traits

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     the

     largest

     city

     in

     the

     country

     and

     the

     seat

     of

     government

    .

     
    


    Paris

    ,

     with

     a

     population

     of

     over

     

    2

     million

     and

     a

     rich

     history

    ,

     is

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

     the

     Lou

    vre

     Museum

    ,

     Notre

     Dame

     Cathedral

    ,

     and

     the

     Palace

     of

     Vers

    ailles

    .

     It

     is

     also

     famous

     for

     its

     French

     cuisine

    ,

     fashion

    ,

     and

     music

    .

     
    


    Paris

     is

     a

     cultural

     melting

     pot

     of

     diverse

     neighborhoods

     and

     is

     home

     to

     many

     famous

     landmarks

    ,

     including

     the

     Lou

    vre

    ,

     Mus

    ée

     d

    '

    Or

    say

    ,

     Mont

    mart

    re

    ,

     and

     the

     Ber

    ling

    ue

    .

     
    


    France

    's

     capital

     city

    ,

     Paris

    ,

     is

     also

     a

     major

     center

     for

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     a

     highly

     dynamic

     and

     rapidly

     evolving

     field

     with

     several

     potential

     trends

     shaping

     its

     development

    .

     Here

     are

     some

     of

     the

     potential

     trends

     that

     could

     impact

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     reliance

     on

     AI

     for

     decision

    -making

     and

     automation

    :

     With

     the

     rise

     of

     automation

     and

     artificial

     intelligence

    ,

     more

     and

     more

     industries

     will

     rely

     on

     AI

    -driven

     decision

    -making

     and

     automation

    .

     This

     could

     lead

     to

     the

     development

     of

     more

     sophisticated

     machine

     learning

     algorithms

     and

     techniques

     that

     can

     make

     more

     accurate

     and

     efficient

     decisions

    .
    


    2

    .

     More

     ethical

     use

     of

     AI

    :

     As

     AI

     technologies

     continue

     to

     develop

    ,

     there

     will

     be

     increased

     scrutiny

     and

     public

     debate

     about

     the

     ethical

     use

     of

     AI

    .

     Governments

     and

     organizations

     will

     need

     to

    



```python
llm.shutdown()
```
