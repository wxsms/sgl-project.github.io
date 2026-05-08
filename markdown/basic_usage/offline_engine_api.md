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

    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Failed to load legacy DeepGEMM A100 Triton kernels: dynamic module does not define module export function (PyInit__C)


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.82it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.81it/s]


    2026-05-08 21:06:49,147 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-08 21:06:49] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:03<03:43,  3.92s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:43,  3.92s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.65it/s]

    Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=1792):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=1536):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=1280):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=1024):  21%|██        | 12/58 [00:04<00:09,  4.90it/s]Compiling num tokens (num_tokens=960):  21%|██        | 12/58 [00:04<00:09,  4.90it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.92it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.92it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.92it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.92it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.92it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.92it/s]Compiling num tokens (num_tokens=576):  38%|███▊      | 22/58 [00:04<00:03, 10.92it/s]Compiling num tokens (num_tokens=512):  38%|███▊      | 22/58 [00:04<00:03, 10.92it/s]Compiling num tokens (num_tokens=480):  38%|███▊      | 22/58 [00:04<00:03, 10.92it/s]Compiling num tokens (num_tokens=448):  38%|███▊      | 22/58 [00:04<00:03, 10.92it/s]

    Compiling num tokens (num_tokens=448):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=416):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=384):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=352):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=320):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=288):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=256):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=240):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=224):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=208):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=192):  53%|█████▎    | 31/58 [00:04<00:01, 17.44it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:04<00:00, 26.05it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:04<00:00, 26.05it/s]Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:04<00:00, 26.05it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:04<00:00, 26.05it/s]Compiling num tokens (num_tokens=128):  71%|███████   | 41/58 [00:04<00:00, 26.05it/s]Compiling num tokens (num_tokens=112):  71%|███████   | 41/58 [00:04<00:00, 26.05it/s]Compiling num tokens (num_tokens=96):  71%|███████   | 41/58 [00:04<00:00, 26.05it/s] Compiling num tokens (num_tokens=80):  71%|███████   | 41/58 [00:04<00:00, 26.05it/s]Compiling num tokens (num_tokens=64):  71%|███████   | 41/58 [00:04<00:00, 26.05it/s]Compiling num tokens (num_tokens=48):  71%|███████   | 41/58 [00:04<00:00, 26.05it/s]Compiling num tokens (num_tokens=32):  71%|███████   | 41/58 [00:04<00:00, 26.05it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:04<00:00, 35.50it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:04<00:00, 35.50it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:04<00:00, 35.50it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:04<00:00, 35.50it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:04<00:00, 35.50it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:04<00:00, 35.50it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:04<00:00, 35.50it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:04<00:00, 35.50it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:04<00:00, 12.59it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=57.95 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.92 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=57.92 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=57.92 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=57.91 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]Capturing num tokens (num_tokens=6144 avail_mem=57.91 GB):   3%|▎         | 2/58 [00:00<00:02, 19.39it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=57.91 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=5632 avail_mem=57.91 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=5120 avail_mem=57.90 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4608 avail_mem=57.90 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.90 GB):   9%|▊         | 5/58 [00:00<00:02, 22.54it/s]Capturing num tokens (num_tokens=4096 avail_mem=57.90 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=3840 avail_mem=57.89 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=3584 avail_mem=56.78 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=56.78 GB):  16%|█▌        | 9/58 [00:00<00:01, 27.33it/s]Capturing num tokens (num_tokens=3328 avail_mem=56.78 GB):  21%|██        | 12/58 [00:00<00:02, 18.15it/s]Capturing num tokens (num_tokens=3072 avail_mem=56.78 GB):  21%|██        | 12/58 [00:00<00:02, 18.15it/s]Capturing num tokens (num_tokens=2816 avail_mem=56.78 GB):  21%|██        | 12/58 [00:00<00:02, 18.15it/s]Capturing num tokens (num_tokens=2560 avail_mem=56.77 GB):  21%|██        | 12/58 [00:00<00:02, 18.15it/s]Capturing num tokens (num_tokens=2304 avail_mem=56.77 GB):  21%|██        | 12/58 [00:00<00:02, 18.15it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=56.77 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=56.77 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.69it/s]Capturing num tokens (num_tokens=1792 avail_mem=56.76 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.69it/s]Capturing num tokens (num_tokens=1536 avail_mem=56.76 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.69it/s]Capturing num tokens (num_tokens=1280 avail_mem=56.76 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.74 GB):  28%|██▊       | 16/58 [00:00<00:01, 22.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=56.74 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=960 avail_mem=56.75 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.32it/s] Capturing num tokens (num_tokens=896 avail_mem=56.75 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=832 avail_mem=56.75 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=768 avail_mem=56.74 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.32it/s]Capturing num tokens (num_tokens=704 avail_mem=56.74 GB):  36%|███▌      | 21/58 [00:00<00:01, 29.32it/s]

    Capturing num tokens (num_tokens=704 avail_mem=56.74 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.47it/s]Capturing num tokens (num_tokens=640 avail_mem=56.74 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.47it/s]Capturing num tokens (num_tokens=576 avail_mem=56.74 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.47it/s]Capturing num tokens (num_tokens=512 avail_mem=56.72 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.47it/s]Capturing num tokens (num_tokens=480 avail_mem=56.74 GB):  45%|████▍     | 26/58 [00:00<00:00, 34.47it/s]Capturing num tokens (num_tokens=448 avail_mem=56.73 GB):  45%|████▍     | 26/58 [00:01<00:00, 34.47it/s]Capturing num tokens (num_tokens=448 avail_mem=56.73 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.40it/s]Capturing num tokens (num_tokens=416 avail_mem=56.73 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.40it/s]Capturing num tokens (num_tokens=384 avail_mem=56.73 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.40it/s]Capturing num tokens (num_tokens=352 avail_mem=56.72 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.40it/s]Capturing num tokens (num_tokens=320 avail_mem=56.72 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.40it/s]Capturing num tokens (num_tokens=288 avail_mem=56.72 GB):  53%|█████▎    | 31/58 [00:01<00:00, 38.40it/s]

    Capturing num tokens (num_tokens=288 avail_mem=56.72 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=256 avail_mem=56.71 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=240 avail_mem=56.71 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=224 avail_mem=56.71 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=208 avail_mem=56.70 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=192 avail_mem=56.70 GB):  62%|██████▏   | 36/58 [00:01<00:00, 41.21it/s]Capturing num tokens (num_tokens=192 avail_mem=56.70 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=176 avail_mem=56.70 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=160 avail_mem=56.70 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=144 avail_mem=56.69 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=128 avail_mem=56.66 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]Capturing num tokens (num_tokens=112 avail_mem=56.65 GB):  71%|███████   | 41/58 [00:01<00:00, 43.40it/s]

    Capturing num tokens (num_tokens=112 avail_mem=56.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.39it/s]Capturing num tokens (num_tokens=96 avail_mem=56.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.39it/s] Capturing num tokens (num_tokens=80 avail_mem=56.65 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.39it/s]Capturing num tokens (num_tokens=64 avail_mem=56.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.39it/s]Capturing num tokens (num_tokens=48 avail_mem=56.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.39it/s]Capturing num tokens (num_tokens=32 avail_mem=56.64 GB):  79%|███████▉  | 46/58 [00:01<00:00, 43.39it/s]Capturing num tokens (num_tokens=32 avail_mem=56.64 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=28 avail_mem=56.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=24 avail_mem=56.63 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=20 avail_mem=56.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=16 avail_mem=56.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.99it/s]Capturing num tokens (num_tokens=12 avail_mem=56.62 GB):  88%|████████▊ | 51/58 [00:01<00:00, 43.99it/s]

    Capturing num tokens (num_tokens=12 avail_mem=56.62 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.20it/s]Capturing num tokens (num_tokens=8 avail_mem=56.62 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.20it/s] Capturing num tokens (num_tokens=4 avail_mem=56.61 GB):  97%|█████████▋| 56/58 [00:01<00:00, 45.20it/s]Capturing num tokens (num_tokens=4 avail_mem=56.61 GB): 100%|██████████| 58/58 [00:01<00:00, 36.08it/s]


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
    Generated text:  Sarah and I'm a high school student. I have a lot of stress and anxiety because I don't know what to do. Is there anything you can suggest for me? I am very worried that I will be depressed and will not be able to learn and do well at school. I have been learning about sleep and study habits and I am trying hard to do it properly. I am very scared of giving up and end up not studying properly. I have always had trouble with my study habits and I am very frustrated. Any advice you can offer me on how to deal with this would be greatly appreciated. Thanks! Sarah
    
    It's
    ===============================
    Prompt: The president of the United States is
    Generated text:  a very important person. He is the leader of the country. He is the boss of the country. He is like a leader in the country. But he is not the only one in the country. There are other important people in the country, too. The other important people are also important in the country. They are the governors of the states. They are like the bosses of the states. They are the head of the government in the states. They are like the bosses in the states. They are the mayor of the city. They are like the bosses in the cities. They are the judges of the courts. They are like
    ===============================
    Prompt: The capital of France is
    Generated text:  _______. Paris Paris Saint-Denis Paris Montmartre Paris
    
    在下面一段话的横线处填入恰当的内容。 有些人认为，英国人过于_________。 有些人认为，英国人过于_________。 有些人认为，英国人过于_________。 这段话有什么问题？
    
    这段话的逻辑关系是：先总后分，符合“有的人……有的人……”的结构。所以，横线处内容应该与“过于”相对应，应该填入“保守”。“有些人认为，英国人过于保守”。这段话表达的是一种看法，所以应该用疑问句。而
    ===============================
    Prompt: The future of AI is
    Generated text:  bright, but the data being used to train these models can be a concern. It’s the kind of thing that got me thinking. As an engineer at a large technology company, I have a responsibility to make the best decisions about how to use data for AI models. So, I spent a few weeks doing a little research into how data can be problematic.
    
    Data bias, specifically:
    
      1. Makes biased predictions
      2. Disproportionate impact, meaning that a particular group of people may get worse treatment than others
      3. Negatively impacts the quality of the training data
    
    What’s a bias?
    
    


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


    Generated text:  Paris, known for its iconic Eiffel Tower, Notre-Dame Cathedral, and vibrant culture. It is also the birthplace of the French Revolution and the home of the French language. Paris is a major cultural and economic center, with a rich history dating back to the Roman Empire and the French Revolution. The city is known for its art, music, and cuisine, and is a popular tourist destination. It is also home to the Louvre Museum, the Eiffel Tower, and the Notre-Dame Cathedral. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly. The city is known for its
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical AI: As more people become aware of the potential risks and biases in AI, there is likely to be a greater focus on ethical AI. This could lead to more stringent regulations and guidelines for AI development and deployment, as well as increased investment in research and development to ensure that AI is used in a responsible and ethical manner.
    
    2. Greater use of AI in healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. As
    


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
    Generated text:  [Your Name] and I am a [type of profession], [Your Profession] with over [number of years of experience] years of experience. I started this [occupation] back in [year] and have always been passionate about [reason for passion] and always strive to [specific goal or accomplishment]. I am a [character trait or quality] who always strives to [specific achievement or goal]. I am always willing to [reason for willingness], and I am always open to learning and taking on new challenges. As a [character role or profession], I am always [character trait or quality] and always strive to [specific accomplishment
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. 
    
    (Note: The answer to the question "What is the capital of France? ") The capital of France is Paris. 
    
    To elaborate, Paris is the capital city of France, situated in the center of the country, surrounded by the Seine River. It is the largest city in France by area, with a population of over 1. 3 million people. Paris is also the world's most populous city, making it the largest city in Europe by population. The city is home to many of France's most famous landmarks, including the Eiffel Tower and Notre-Dame Cathedral. 
    
    Paris is known for its rich
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be heavily influenced by advancements in the fields of machine learning, deep learning, and quantum computing. Here are some possible future trends in AI:
    
    1. Increased Human-Computer Interaction: As AI becomes more capable and efficient, it is likely to be able to perform tasks that were previously done by humans, such as decision-making, creativity, and language translation. This could lead to increased human-computer interaction and a more immersive digital experience.
    
    2. More Customizable AI: As AI systems become more capable and accurate, they may be able to learn from the interactions with users and become more personalized. This could lead to more customized AI


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

    insert

     name

    ].

     I

     am

     an

     experienced

     professional

     with

     a

     keen

     interest

     in

     technology

    ,

     writing

    ,

     and

     storytelling

    .

     I

     love

     to

     create

     unique

     and

     compelling

     stories

     that

     push

     the

     boundaries

     of

     what

    's

     possible

    ,

     whether

     it

    's

     fiction

     or

     non

    -fiction

    .

     I

    'm

     a

     firm

     believer

     in

     the

     importance

     of

     creativity

     and

     imagination

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     push

     the

     envelope

     and

     create

     something

     new

     and

     exciting

    .

     I

     believe

     that

     the

     key

     to

     success

     in

     any

     field

     is

     to

     stay

     curious

     and

     keep

     learning

    ,

     and

     I

    'm

     here

     to

     do

     just

     that

    !

     [

    insert

     name

    ]

     is

     passionate

     about

     sharing

     my

     ideas

     and

     experiences

    ,

     and

     I

    'm

     always

     eager

     to

     learn

     and

     grow

     with

     you

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .
    


    Is

     the

     following

     a

     factual

     statement

    ?


    "

    La

     La

     Land

     stars

     French

     actor

     Robert

     De

     N

    iro

    ."


    Options

     are

    :


    1

    ).

     yes

    


    2

    ).

     no

     

    2

    ).

     no

    
    


    La

     La

     Land

     stars

     French

     actor

     Robert

     De

     N

    iro

    .

     Robert

     De

     N

    iro

     was

     the

     lead

     actor

     in

     the

     movie

    ,

     which

     was

     directed

     by

     Damien

     Ch

    az

    elle

     and

     stars

     a

     young

     actress

     named

     Ch

    lo

    é

     Zhu

    .

     The

     movie

     is

     a

     musical

     romance

     about

     two

     young

     women

     who

     fall

     in

     love

     through

     a

     computer

     game

    .

     Robert

     De

     N

    iro

     plays

     the

     character

     of

     a

     struggling

     musician

     who

     becomes

     an

     internet

     sensation

     after

     he

     creates

     a

     revolutionary

     online

     game

     called

     La

     La

     Land

    .

     However

    ,

     Robert

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     involve

     a

     combination

     of

     rapid

     progress

    ,

     breakthrough

    s

    ,

     and

     challenges

    .

     Here

     are

     some

     possible

     trends

    :
    


    1

    .

     Autonomous

     vehicles

    :

     AI

     is

     already

     revolution

    izing

     the

     transportation

     industry

    ,

     and

     the

     future

     of

     AI

     could

     see

     autonomous

     vehicles

     becoming

     more

     common

    .

     These

     vehicles

     would

     be

     able

     to

     navigate

     roads

    ,

     stop

    ,

     turn

    ,

     and

     even

     make

     decisions

     on

     their

     own

     based

     on

     sensors

     and

     computer

     algorithms

    .
    


    2

    .

     Smart

     homes

    :

     AI

     is

     being

     used

     in

     the

     development

     of

     smart

     homes

    ,

     which

     could

     include

     automated

     lighting

    ,

     temperature

     control

    ,

     and

     security

     systems

    .

     These

     systems

     would

     be

     able

     to

     learn

     from

     users

    '

     behavior

     and

     make

     adjustments

     as

     needed

    .
    


    3

    .

     Personal

    ized

     healthcare

    :

     AI

    



```python
llm.shutdown()
```
