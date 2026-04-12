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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.39it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  7.37it/s]


    2026-04-12 08:56:32,194 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-12 08:56:32] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:02<02:34,  2.71s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]

    Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:02<00:22,  2.33it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:02<00:10,  4.74it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:02<00:10,  4.74it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:02<00:10,  4.74it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:03<00:10,  4.74it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:03<00:10,  4.74it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:03<00:05,  7.68it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:03<00:05,  7.68it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:03<00:05,  7.68it/s]Compiling num tokens (num_tokens=2304):  22%|██▏       | 13/58 [00:03<00:05,  7.68it/s]

    Compiling num tokens (num_tokens=2048):  22%|██▏       | 13/58 [00:03<00:05,  7.68it/s]Compiling num tokens (num_tokens=1792):  22%|██▏       | 13/58 [00:03<00:05,  7.68it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.19it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.19it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.19it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.19it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.19it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.19it/s]Compiling num tokens (num_tokens=896):  40%|███▉      | 23/58 [00:03<00:02, 17.14it/s]Compiling num tokens (num_tokens=832):  40%|███▉      | 23/58 [00:03<00:02, 17.14it/s]Compiling num tokens (num_tokens=768):  40%|███▉      | 23/58 [00:03<00:02, 17.14it/s]Compiling num tokens (num_tokens=704):  40%|███▉      | 23/58 [00:03<00:02, 17.14it/s]

    Compiling num tokens (num_tokens=640):  40%|███▉      | 23/58 [00:03<00:02, 17.14it/s]Compiling num tokens (num_tokens=576):  40%|███▉      | 23/58 [00:03<00:02, 17.14it/s]Compiling num tokens (num_tokens=512):  40%|███▉      | 23/58 [00:03<00:02, 17.14it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:03<00:01, 23.23it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 29.33it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 29.33it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 29.33it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 29.33it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 29.33it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 29.33it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:03<00:00, 32.79it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:03<00:00, 32.79it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:03<00:00, 32.79it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:03<00:00, 32.79it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:03<00:00, 32.79it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:03<00:00, 32.79it/s]Compiling num tokens (num_tokens=128):  78%|███████▊  | 45/58 [00:03<00:00, 36.13it/s]Compiling num tokens (num_tokens=112):  78%|███████▊  | 45/58 [00:03<00:00, 36.13it/s]Compiling num tokens (num_tokens=96):  78%|███████▊  | 45/58 [00:03<00:00, 36.13it/s] 

    Compiling num tokens (num_tokens=80):  78%|███████▊  | 45/58 [00:03<00:00, 36.13it/s]Compiling num tokens (num_tokens=64):  78%|███████▊  | 45/58 [00:03<00:00, 36.13it/s]Compiling num tokens (num_tokens=48):  78%|███████▊  | 45/58 [00:03<00:00, 36.13it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 39.04it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 39.04it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 39.04it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 39.04it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 39.04it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 39.04it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 39.04it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 39.04it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 39.04it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 14.81it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=118.09 GB):   2%|▏         | 1/58 [00:00<00:06,  8.93it/s]Capturing num tokens (num_tokens=7680 avail_mem=118.06 GB):   2%|▏         | 1/58 [00:00<00:06,  8.93it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=118.06 GB):   2%|▏         | 1/58 [00:00<00:06,  8.93it/s]Capturing num tokens (num_tokens=7168 avail_mem=118.06 GB):   5%|▌         | 3/58 [00:00<00:05, 10.59it/s]Capturing num tokens (num_tokens=6656 avail_mem=118.06 GB):   5%|▌         | 3/58 [00:00<00:05, 10.59it/s]Capturing num tokens (num_tokens=6144 avail_mem=118.06 GB):   5%|▌         | 3/58 [00:00<00:05, 10.59it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=118.06 GB):   9%|▊         | 5/58 [00:00<00:04, 11.99it/s]Capturing num tokens (num_tokens=5632 avail_mem=118.05 GB):   9%|▊         | 5/58 [00:00<00:04, 11.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.06 GB):   9%|▊         | 5/58 [00:00<00:04, 11.99it/s]Capturing num tokens (num_tokens=5120 avail_mem=118.06 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.71it/s]Capturing num tokens (num_tokens=4608 avail_mem=118.05 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=118.05 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=118.05 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.71it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=118.04 GB):  12%|█▏        | 7/58 [00:00<00:03, 13.71it/s]Capturing num tokens (num_tokens=3584 avail_mem=118.04 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=3328 avail_mem=118.04 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=3072 avail_mem=118.04 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=2816 avail_mem=118.04 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.55it/s]Capturing num tokens (num_tokens=2560 avail_mem=118.03 GB):  19%|█▉        | 11/58 [00:00<00:02, 20.55it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=118.03 GB):  26%|██▌       | 15/58 [00:00<00:02, 18.14it/s]Capturing num tokens (num_tokens=2304 avail_mem=118.92 GB):  26%|██▌       | 15/58 [00:00<00:02, 18.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.91 GB):  26%|██▌       | 15/58 [00:00<00:02, 18.14it/s]Capturing num tokens (num_tokens=2048 avail_mem=118.91 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.59it/s]Capturing num tokens (num_tokens=1792 avail_mem=137.35 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.59it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=137.34 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.59it/s]Capturing num tokens (num_tokens=1280 avail_mem=137.34 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.59it/s]Capturing num tokens (num_tokens=1024 avail_mem=137.32 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.59it/s]Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  29%|██▉       | 17/58 [00:01<00:02, 15.59it/s] Capturing num tokens (num_tokens=960 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.22it/s]Capturing num tokens (num_tokens=896 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.22it/s]Capturing num tokens (num_tokens=832 avail_mem=137.33 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.22it/s]Capturing num tokens (num_tokens=768 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.22it/s]Capturing num tokens (num_tokens=704 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.22it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  38%|███▊      | 22/58 [00:01<00:01, 22.22it/s]Capturing num tokens (num_tokens=640 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.83it/s]Capturing num tokens (num_tokens=576 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.83it/s]

    Capturing num tokens (num_tokens=512 avail_mem=137.30 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.83it/s]Capturing num tokens (num_tokens=480 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.83it/s]Capturing num tokens (num_tokens=448 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.83it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  47%|████▋     | 27/58 [00:01<00:01, 27.83it/s]Capturing num tokens (num_tokens=416 avail_mem=137.32 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.29it/s]Capturing num tokens (num_tokens=384 avail_mem=137.31 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.29it/s]Capturing num tokens (num_tokens=352 avail_mem=137.31 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.29it/s]Capturing num tokens (num_tokens=320 avail_mem=137.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.29it/s]Capturing num tokens (num_tokens=288 avail_mem=137.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.29it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  55%|█████▌    | 32/58 [00:01<00:00, 32.29it/s]Capturing num tokens (num_tokens=256 avail_mem=137.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=240 avail_mem=137.30 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.65it/s]

    Capturing num tokens (num_tokens=224 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=208 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=192 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  64%|██████▍   | 37/58 [00:01<00:00, 35.65it/s]Capturing num tokens (num_tokens=176 avail_mem=137.29 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=160 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=144 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=128 avail_mem=137.28 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=112 avail_mem=137.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.34it/s]Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  72%|███████▏  | 42/58 [00:01<00:00, 38.34it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=137.27 GB):  81%|████████  | 47/58 [00:01<00:00, 39.35it/s]Capturing num tokens (num_tokens=80 avail_mem=137.27 GB):  81%|████████  | 47/58 [00:01<00:00, 39.35it/s]Capturing num tokens (num_tokens=64 avail_mem=137.26 GB):  81%|████████  | 47/58 [00:01<00:00, 39.35it/s]Capturing num tokens (num_tokens=48 avail_mem=137.26 GB):  81%|████████  | 47/58 [00:01<00:00, 39.35it/s]Capturing num tokens (num_tokens=32 avail_mem=137.25 GB):  81%|████████  | 47/58 [00:01<00:00, 39.35it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  81%|████████  | 47/58 [00:01<00:00, 39.35it/s]Capturing num tokens (num_tokens=28 avail_mem=137.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.81it/s]Capturing num tokens (num_tokens=24 avail_mem=137.25 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.81it/s]Capturing num tokens (num_tokens=20 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.81it/s]Capturing num tokens (num_tokens=16 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.81it/s]Capturing num tokens (num_tokens=12 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.81it/s]Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  90%|████████▉ | 52/58 [00:01<00:00, 40.81it/s] 

    Capturing num tokens (num_tokens=8 avail_mem=137.24 GB):  98%|█████████▊| 57/58 [00:02<00:00, 42.30it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB):  98%|█████████▊| 57/58 [00:02<00:00, 42.30it/s]Capturing num tokens (num_tokens=4 avail_mem=137.23 GB): 100%|██████████| 58/58 [00:02<00:00, 28.64it/s]


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
    Generated text:  HelloWorm and I am a fully customizable and fully functional character from the popular video game "Zelda." I'm also a character from the game "Pokémon."
    
    What other games do I know or have played before? Yes, I have played the classic video game "Zelda." It's a role-playing game about a young girl named Zelda who discovers the hidden world of the Enchanted Forest, a mystical realm filled with dragons, trolls, and other fantastical creatures.
    
    How would I like to play a different game? That sounds interesting! How can I customize my character for that game? That's great! Customizing your
    ===============================
    Prompt: The president of the United States is
    Generated text:  invited to a special dinner. There are 10 seats at a table in the United States. He is seated at a random table, but only if there is at least one other person sitting. If there are no other people, he will be at the head of the table. If there are more than 2 people sitting, he will be seated at the second chair from the head. If there is only one person sitting, he will be at the head of the table. If all the seating arrangements are equally likely, what is the probability that the president will be at the head of the table?
    To determine the probability that the
    ===============================
    Prompt: The capital of France is
    Generated text:  [ ]
    A. Paris
    B. Lille
    C. Lyon
    D. Marseille
    Answer:
    
    A
    
    Which of the following statements about the Hainan International Horticultural Expo is incorrect?
    A. The Hainan International Horticultural Expo is held every two years.
    B. It is a major international event promoting the development of horticulture in China.
    C. It is the world's largest horticultural exhibition.
    D. The Expo lasts from September 30 to October 22.
    Answer:
    
    D
    
    Which of the following is NOT a reason why the land in a certain region is susceptible to desertification?
    
    ===============================
    Prompt: The future of AI is
    Generated text:  digital, which will make it an integral part of every industry. Many industries are looking to develop new AI technologies in order to operate more efficiently. The ability to work with AI is great for the future as it will create opportunities for increased efficiency in the industry.
    One of the most effective ways to improve an industry is to create new AI technologies that can solve complex problems. Many industries are looking for new solutions to existing problems, and AI can help with this. AI can be used in many different ways, including in healthcare, transportation, and manufacturing.
    AI is an integral part of any industry, and it can be used to improve efficiency,


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


    Generated text:  Paris. 
    
    The statement is concise and accurately describes the capital city of France. It provides the name of the city and its capital, which are the key pieces of information needed to understand the statement. The statement is clear and easy to understand, making it suitable for a variety of contexts. Additionally, it is specific enough to be useful in a broader discussion about France's political and cultural landscape. Overall, the statement is a good representation of the capital city of France. 
    
    The statement is also accurate, as it correctly identifies Paris as the capital city of France. It does not make any assumptions or omissions, and provides a clear and
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by rapid advancements in areas such as machine learning, natural language processing, and computer vision. These technologies will continue to improve, leading to more sophisticated and accurate AI systems that can perform a wide range of tasks, from simple tasks like language translation to complex tasks like autonomous driving and medical diagnosis. As AI becomes more integrated into our daily lives, we can expect to see even more widespread adoption of AI in areas such as healthcare, finance, and transportation. Additionally, AI will continue to be used for research and development, with the goal of further advancing our understanding of the world and developing new technologies. Finally, AI will continue
    


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
    Generated text:  [Name] and I'm a/an [age] year old [career], [job title], [position]. I enjoy [reason for interest in your field], [speciality]. I'm an [occupation], [personality trait]. I'm friendly, funny, and [your favorite hobby or interest]. I'm a [city] native. I love [sport], [diet], or [activity]. I have a passion for [field], [personality, or interest], and I'm always looking for new experiences and ways to grow personally and professionally. Thank you! [Name]. Your self-introduction is clear and concise.
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a bustling metropolis that is home to a diverse population of around 2.3 million people and is renowned for its rich history, culture, and modern-day vibrant city life. Paris is a UNESCO World Heritage site and is the world’s largest city, located on the banks of the Seine River, and is home to the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. It is also known for its iconic architecture, including the Notre-Dame Cathedral, the Louvre, the Palais Royal, and the Arc de Triomphe. Paris is a cultural hub, known for its renowned
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to continue to expand, and there are many possible trends to watch for. Here are some possible trends that could shape the future of AI:
    
    1. Increased use of AI in healthcare: AI can help doctors diagnose diseases, predict patient outcomes, and optimize treatment plans. It could also help in developing more personalized treatments, reducing the risk of medication side effects, and improving patient outcomes.
    
    2. Enhanced natural language processing: With advancements in AI, it is expected that we will see more accurate and efficient natural language processing. This will enable machines to understand human language and communicate more effectively, improve personalization of services, and automate mundane tasks.
    
    


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

    job

     title

    ]

     with

     over

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

     I

     work

     hard

     to

     improve

     my

     skills

     and

     ensure

     that

     I

     can

     provide

     the

     best

     service

     possible

     to

     our

     customers

    .

     I

    'm

     always

     up

     for

     learning

     new

     things

     and

     looking

     for

     ways

     to

     stay

     ahead

     of

     the

     game

    .

     I

    'm

     excited

     to

     work

     with

     everyone

     at

     [

    company

     name

    ]

     and

     make

     sure

     that

     everyone

     is

     happy

    .

     Looking

     forward

     to

     meeting

     you

    !

     [

    Your

     Name

    ].

     

    📊

    💼

    💼

     #

    Meet

    The

    Hero

     

    🧵

    ✨

     #

    Friendly

    Chat

    


    **

    Neutral

     Self

    -

    Introduction

     for

     a

     Fiction

    al

     Character

    **
    


    Hello

    ,

     my

     name

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    .

     It

     is

     the

     largest

     city

     in

     the

     country

     and

     is

     known

     for

     its

     rich

     history

    ,

     beautiful

     architecture

    ,

     and

     vibrant

     culture

    .

     Paris

     is

     also

     the

     birth

    place

     of

     many

     influential

     figures

     in

     world

     history

     and

     is

     recognized

     as

     a

     global

     cultural

     capital

    .

     The

     city

     is

     home

     to

     many

     museums

    ,

     monuments

    ,

     and

     theaters

    ,

     and

     is

     a

     major

     hub

     for

     global

     trade

    ,

     finance

    ,

     and

     culture

    .

     The

     city

     is

     also

     home

     to

     numerous

     famous

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

    -D

    ame

     Cathedral

    ,

     and

     Mont

    mart

    re

    .

     Paris

     is

     a

     bustling

     and

     exciting

     city

     with

     a

     rich

     history

     and

     diverse

     culture

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

     also

     one

     of

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     be

     highly

     interconnected

     with

     the

     world

    's

     rapidly

     growing

     digital

     landscape

    .

     It

     is

     likely

     that

     AI

     will

     continue

     to

     evolve

     and

     expand

     in

     ways

     that

     are

     both

     exciting

     and

     challenging

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

     Increased

     development

     of

     AI

    -powered

     healthcare

     systems

    :

     As

     AI

     becomes

     more

     sophisticated

    ,

     it

     will

     be

     able

     to

     help

     healthcare

     providers

     diagnose

     diseases

    ,

     monitor

     patients

    ,

     and

     develop

     personalized

     treatment

     plans

    .

     This

     could

     lead

     to

     more

     efficient

     and

     effective

     use

     of

     resources

     in

     the

     healthcare

     system

    .
    


    2

    .

     Increased

     integration

     of

     AI

     into

     consumer

     products

    :

     AI

     is

     already

     being

     used

     in

     consumer

     products

     like

     smart

     home

     devices

    ,

     self

    -driving

     cars

    ,

     and

     virtual

     reality

     experiences

    .

     As

    



```python
llm.shutdown()
```
