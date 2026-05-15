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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.16it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.15it/s]


    2026-05-15 03:10:59,884 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-15 03:10:59] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:26,  4.67s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]Compiling num tokens (num_tokens=6144):   5%|▌         | 3/58 [00:04<01:09,  1.26s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:33,  1.56it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:05<00:16,  3.06it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:05<00:16,  3.06it/s]Compiling num tokens (num_tokens=3840):  14%|█▍        | 8/58 [00:05<00:16,  3.06it/s]Compiling num tokens (num_tokens=3584):  14%|█▍        | 8/58 [00:05<00:16,  3.06it/s]

    Compiling num tokens (num_tokens=3328):  14%|█▍        | 8/58 [00:05<00:16,  3.06it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:05<00:08,  5.63it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:05<00:08,  5.63it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:05<00:08,  5.63it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:05<00:08,  5.63it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:05<00:08,  5.63it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:05<00:08,  5.63it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:05<00:04,  9.65it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:05<00:04,  9.65it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:05<00:04,  9.65it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:05<00:04,  9.65it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:05<00:04,  9.65it/s]

    Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:05<00:04,  9.65it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:05<00:04,  9.65it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:05<00:02, 15.23it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:05<00:02, 15.23it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:05<00:02, 15.23it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:05<00:02, 15.23it/s]Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:05<00:02, 15.23it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:05<00:02, 15.23it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:05<00:02, 15.23it/s]Compiling num tokens (num_tokens=480):  40%|███▉      | 23/58 [00:05<00:02, 15.23it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:05<00:01, 22.45it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:05<00:01, 22.45it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:05<00:01, 22.45it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:05<00:01, 22.45it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:05<00:01, 22.45it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:05<00:01, 22.45it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:05<00:01, 22.45it/s]Compiling num tokens (num_tokens=256):  52%|█████▏    | 30/58 [00:05<00:01, 22.45it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=144):  64%|██████▍   | 37/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=128):  64%|██████▍   | 37/58 [00:05<00:00, 29.83it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:05<00:00, 38.20it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:05<00:00, 38.20it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:05<00:00, 38.20it/s] Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:05<00:00, 38.20it/s]

    Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:05<00:00, 38.20it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:05<00:00, 38.20it/s]Compiling num tokens (num_tokens=32):  78%|███████▊  | 45/58 [00:05<00:00, 38.20it/s]Compiling num tokens (num_tokens=28):  78%|███████▊  | 45/58 [00:05<00:00, 38.20it/s]Compiling num tokens (num_tokens=24):  78%|███████▊  | 45/58 [00:05<00:00, 38.20it/s]Compiling num tokens (num_tokens=24):  91%|█████████▏| 53/58 [00:05<00:00, 46.04it/s]Compiling num tokens (num_tokens=20):  91%|█████████▏| 53/58 [00:05<00:00, 46.04it/s]Compiling num tokens (num_tokens=16):  91%|█████████▏| 53/58 [00:05<00:00, 46.04it/s]Compiling num tokens (num_tokens=12):  91%|█████████▏| 53/58 [00:05<00:00, 46.04it/s]Compiling num tokens (num_tokens=8):  91%|█████████▏| 53/58 [00:05<00:00, 46.04it/s] Compiling num tokens (num_tokens=4):  91%|█████████▏| 53/58 [00:05<00:00, 46.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00,  9.89it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=50.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=50.67 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=50.67 GB):   3%|▎         | 2/58 [00:00<00:05, 10.03it/s]Capturing num tokens (num_tokens=7168 avail_mem=50.67 GB):   3%|▎         | 2/58 [00:00<00:05, 10.03it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=50.67 GB):   3%|▎         | 2/58 [00:00<00:05, 10.03it/s]Capturing num tokens (num_tokens=6656 avail_mem=50.67 GB):   7%|▋         | 4/58 [00:00<00:05, 10.75it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.67 GB):   7%|▋         | 4/58 [00:00<00:05, 10.75it/s]Capturing num tokens (num_tokens=5632 avail_mem=50.66 GB):   7%|▋         | 4/58 [00:00<00:05, 10.75it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=50.66 GB):  10%|█         | 6/58 [00:00<00:04, 11.75it/s]Capturing num tokens (num_tokens=5120 avail_mem=50.65 GB):  10%|█         | 6/58 [00:00<00:04, 11.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=50.65 GB):  10%|█         | 6/58 [00:00<00:04, 11.75it/s]Capturing num tokens (num_tokens=4608 avail_mem=50.65 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.03it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.65 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.03it/s]Capturing num tokens (num_tokens=3840 avail_mem=50.65 GB):  14%|█▍        | 8/58 [00:00<00:03, 13.03it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=50.65 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.51it/s]Capturing num tokens (num_tokens=3584 avail_mem=50.64 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.51it/s]Capturing num tokens (num_tokens=3328 avail_mem=50.64 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.63 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.51it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.63 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.81it/s]Capturing num tokens (num_tokens=2816 avail_mem=50.63 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.81it/s]Capturing num tokens (num_tokens=2560 avail_mem=50.63 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.81it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=50.63 GB):  22%|██▏       | 13/58 [00:00<00:02, 16.81it/s]Capturing num tokens (num_tokens=2304 avail_mem=50.63 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.07it/s]Capturing num tokens (num_tokens=2048 avail_mem=50.62 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=50.62 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=50.62 GB):  28%|██▊       | 16/58 [00:01<00:02, 19.07it/s]Capturing num tokens (num_tokens=1536 avail_mem=50.62 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.17it/s]Capturing num tokens (num_tokens=1280 avail_mem=50.61 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.17it/s]Capturing num tokens (num_tokens=1024 avail_mem=50.59 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.17it/s]

    Capturing num tokens (num_tokens=960 avail_mem=50.61 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.17it/s] Capturing num tokens (num_tokens=896 avail_mem=50.61 GB):  33%|███▎      | 19/58 [00:01<00:01, 21.17it/s]Capturing num tokens (num_tokens=896 avail_mem=50.61 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.81it/s]Capturing num tokens (num_tokens=832 avail_mem=50.60 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.81it/s]Capturing num tokens (num_tokens=768 avail_mem=50.60 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.81it/s]Capturing num tokens (num_tokens=704 avail_mem=50.60 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.81it/s]Capturing num tokens (num_tokens=640 avail_mem=50.59 GB):  40%|███▉      | 23/58 [00:01<00:01, 24.81it/s]Capturing num tokens (num_tokens=640 avail_mem=50.59 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.44it/s]Capturing num tokens (num_tokens=576 avail_mem=50.59 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.44it/s]

    Capturing num tokens (num_tokens=512 avail_mem=50.58 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.44it/s]Capturing num tokens (num_tokens=480 avail_mem=50.59 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.44it/s]Capturing num tokens (num_tokens=448 avail_mem=50.59 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.44it/s]Capturing num tokens (num_tokens=448 avail_mem=50.59 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.26it/s]Capturing num tokens (num_tokens=416 avail_mem=50.59 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.26it/s]Capturing num tokens (num_tokens=384 avail_mem=50.59 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.26it/s]Capturing num tokens (num_tokens=352 avail_mem=50.58 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.26it/s]Capturing num tokens (num_tokens=320 avail_mem=50.57 GB):  53%|█████▎    | 31/58 [00:01<00:00, 29.26it/s]

    Capturing num tokens (num_tokens=320 avail_mem=50.57 GB):  60%|██████    | 35/58 [00:01<00:00, 31.01it/s]Capturing num tokens (num_tokens=288 avail_mem=50.57 GB):  60%|██████    | 35/58 [00:01<00:00, 31.01it/s]Capturing num tokens (num_tokens=256 avail_mem=50.57 GB):  60%|██████    | 35/58 [00:01<00:00, 31.01it/s]Capturing num tokens (num_tokens=240 avail_mem=50.57 GB):  60%|██████    | 35/58 [00:01<00:00, 31.01it/s]Capturing num tokens (num_tokens=224 avail_mem=50.56 GB):  60%|██████    | 35/58 [00:01<00:00, 31.01it/s]Capturing num tokens (num_tokens=224 avail_mem=50.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.46it/s]Capturing num tokens (num_tokens=208 avail_mem=50.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.46it/s]Capturing num tokens (num_tokens=192 avail_mem=50.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.46it/s]Capturing num tokens (num_tokens=176 avail_mem=50.56 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.46it/s]Capturing num tokens (num_tokens=160 avail_mem=50.55 GB):  67%|██████▋   | 39/58 [00:01<00:00, 32.46it/s]

    Capturing num tokens (num_tokens=160 avail_mem=50.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.49it/s]Capturing num tokens (num_tokens=144 avail_mem=50.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.49it/s]Capturing num tokens (num_tokens=128 avail_mem=50.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.49it/s]Capturing num tokens (num_tokens=112 avail_mem=50.55 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.49it/s]Capturing num tokens (num_tokens=96 avail_mem=50.54 GB):  74%|███████▍  | 43/58 [00:01<00:00, 33.49it/s] Capturing num tokens (num_tokens=96 avail_mem=50.54 GB):  81%|████████  | 47/58 [00:01<00:00, 32.68it/s]Capturing num tokens (num_tokens=80 avail_mem=50.54 GB):  81%|████████  | 47/58 [00:01<00:00, 32.68it/s]Capturing num tokens (num_tokens=64 avail_mem=50.53 GB):  81%|████████  | 47/58 [00:01<00:00, 32.68it/s]Capturing num tokens (num_tokens=48 avail_mem=50.53 GB):  81%|████████  | 47/58 [00:02<00:00, 32.68it/s]

    Capturing num tokens (num_tokens=32 avail_mem=50.53 GB):  81%|████████  | 47/58 [00:02<00:00, 32.68it/s]Capturing num tokens (num_tokens=32 avail_mem=50.53 GB):  88%|████████▊ | 51/58 [00:02<00:00, 34.52it/s]Capturing num tokens (num_tokens=28 avail_mem=50.52 GB):  88%|████████▊ | 51/58 [00:02<00:00, 34.52it/s]Capturing num tokens (num_tokens=24 avail_mem=50.52 GB):  88%|████████▊ | 51/58 [00:02<00:00, 34.52it/s]Capturing num tokens (num_tokens=20 avail_mem=50.52 GB):  88%|████████▊ | 51/58 [00:02<00:00, 34.52it/s]Capturing num tokens (num_tokens=16 avail_mem=50.52 GB):  88%|████████▊ | 51/58 [00:02<00:00, 34.52it/s]Capturing num tokens (num_tokens=16 avail_mem=50.52 GB):  95%|█████████▍| 55/58 [00:02<00:00, 31.69it/s]Capturing num tokens (num_tokens=12 avail_mem=50.51 GB):  95%|█████████▍| 55/58 [00:02<00:00, 31.69it/s]Capturing num tokens (num_tokens=8 avail_mem=50.51 GB):  95%|█████████▍| 55/58 [00:02<00:00, 31.69it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=50.51 GB):  95%|█████████▍| 55/58 [00:02<00:00, 31.69it/s]Capturing num tokens (num_tokens=4 avail_mem=50.51 GB): 100%|██████████| 58/58 [00:02<00:00, 25.42it/s]


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
    Generated text:  Eliza. I'm a graphic designer based in Sydney. I'm excited to meet you here at the Sydney design conference. I'm in Sydney as part of a team and there is a lot of work to do for the next few days.
    Today we're going to be looking at a series of images from a book by Robert Mapplethorpe. The book is called The Kid in the Hat. In this book, Mapplethorpe painted an illustration of a little boy being carried around by a very large dog. In this series of images, we see different angles and perspectives of this illustration. I'm going to show
    ===============================
    Prompt: The president of the United States is
    Generated text:  a member of the Senate and the president of the Senate is the president of which body?  Answer the above question. (There is only one correct answer.)
    The answer to this question is:
    Senate of the United States The president of the United States is a member of the Senate, and the president of the Senate is the president of the United States. The Senate is the upper chamber of the United States Congress, and it has the power to confirm the qualifications for being president, including qualifications for membership in the Senate itself. The president of the Senate, who is often referred to as the "chief of the Senate", has the power to
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. It was founded as the capital of the Latin Kingdom of Paris (11th century) and is the capital of the Department of Paris. The city is located at the center of the Mediterranean Sea, between the United Kingdom and the United States, and is one of the most populous cities in Europe. Paris is the capital of the Fifth Region of France and the 10th largest metropolitan area of France.
    
    The city of Paris is the capital of France. Its most well known landmark is Notre-Dame Cathedral, which is a Gothic style building located in the middle of the heart of Paris. Its heart is surrounded by the Luxembourg
    ===============================
    Prompt: The future of AI is
    Generated text:  moving quickly. Advancements in artificial intelligence and machine learning have made it possible for AI to be applied to many different areas, including healthcare, education, and finance. As such, it is becoming more and more important to understand how AI can be used to improve people’s lives.
    
    One of the most promising areas for AI is in healthcare. With the increasing availability of data from various sources, including electronic health records and clinical records, AI can be used to improve the accuracy and efficiency of diagnosis and treatment. For example, AI can analyze large amounts of medical data to identify patterns and predict disease outcomes, allowing healthcare providers to make more informed decisions


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a brief description of your profession or skills]. I enjoy [insert a short description of your hobbies or interests]. What's your favorite hobby or activity? I love [insert a short description of your favorite activity]. What's your favorite book or movie? I love [insert a short description of your favorite book or movie]. What's your favorite place to go? I love [insert a short description of your favorite place]. What's
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a bustling metropolis with a rich cultural heritage and is a major tourist destination. The city is also known for its cuisine, including its famous croissants and its traditional French wine. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a city that has played a significant role in French history and culture, and continues to be
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn from and adapt to human behavior and decision-making processes. This could lead to more efficient and effective AI systems that can better understand and respond to human needs and preferences.
    
    2. Enhanced privacy and security: As AI systems become more integrated with human intelligence, there will be increased concerns about privacy and security. There will be a need for more robust privacy and security measures to protect the data and information that is generated and used by AI systems.
    
    3. Greater emphasis on ethical AI: As
    


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
    Generated text:  [First Name] and I am [Last Name]. I love [mention a hobby or passion you have] and I enjoy [mention a particular skill or area of interest]. I'm a [mention any significant achievement or accomplishment you've made]. I also enjoy [mention a hobby or interest in sports, music, or art]. I believe that my [mention a personal trait or value you hold dear] is essential for me to succeed in life. I'm a [mention any profession, hobby, or area of interest you are passionate about]. I love to [mention a particular activity or hobby]. I am excited to [mention any upcoming
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, known for its iconic landmarks, diverse culture, and rich history.
    
    Key points:
    
    - Paris is the capital city of France and is also known as the "City of Love."
    - It is home to the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and many other famous landmarks.
    - Paris is known for its romantic atmosphere, romantic romance, and love stories.
    - It is home to some of the world's most famous museums, including the Louvre and the Musée d'Orsay.
    - Paris is a major hub for international trade, finance, and culture.
    - It is a cosmopolitan city
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  likely to be characterized by continued growth, development, and adoption across various industries and domains. Here are some potential future trends in AI:
    
    1. Increased use of AI in healthcare: AI is already being used to assist in diagnosing diseases and managing patient care. As AI continues to improve and becomes more accessible, we can expect to see a growing role for AI in healthcare, from developing new treatments to improving the efficiency of medical practice.
    
    2. Enhanced personalization of AI: As AI continues to learn from user data and interactions, it will become increasingly personalized. This will help to improve the accuracy of predictions and recommendations, as well as provide


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

    'm

     a

     [

    occupation

    ].

     I

     was

     born

     and

     raised

     in

     [

    country

    ],

     and

     I

    've

     always

     been

     passionate

     about

     [

    occupation

    ].

     I

     enjoy

     [

    occupation

    ]

     because

     it

     allows

     me

     to

     [

    what

     you

     did

     in

     previous

     jobs

    /

    education

    ],

     and

     I

     believe

     in

     [

    occupation

    ]

     because

     it

     helps

     me

     [

    what

     you

     believe

     in

     about

     your

     profession

    ].


    I

     have

     always

     been

     a

     [

    description

     of

     your

     professional

    /

    character

     traits

    ],

     and

     I

     am

     always

     looking

     for

     [

    how

     you

     can

     improve

     yourself

    ].

     What

     are

     your

     goals

     for

     the

     future

    ,

     and

     what

     are

     you

     looking

     forward

     to

     achieving

    ?


    You

     might

     also

     like

    :


    1

    .

     "

    Hello

    ,

     my

     name

     is

     [

    name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     very

     promising

    ,

     with

     exciting

     new

     trends

     and

     innovations

     making

     it

     all

     the

     more

     exciting

     to

     watch

    .

     Here

     are

     some

     potential

     future

     trends

     in

     AI

    :
    


    1

    .

     AI

     Ethics

     and

     Transparency

    :

     One

     of

     the

     biggest

     challenges

     facing

     AI

     is

     ensuring

     that

     it

     is

     developed

     and

     used

     eth

    ically

     and

     transparent

    ly

    .

     As

     more

     AI

     systems

     become

     integrated

     into

     our

     daily

     lives

    ,

     it

     is

     becoming

     increasingly

     important

     to

     establish

     clear

     guidelines

     for

     how

     AI

     is

     developed

    ,

     deployed

    ,

     and

     used

    .
    


    2

    .

     AI

     in

     Healthcare

    :

     AI

     is

     already

     being

     used

     to

     improve

     the

     accuracy

     and

     speed

     of

     medical

     diagnosis

    ,

     and

     in

     the

     future

    ,

     it

     may

     be

     used

     to

     develop

     new

     treatments

     for

     diseases

    .

     However

    ,

     it

     is

     important

    



```python
llm.shutdown()
```
