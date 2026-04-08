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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.23it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.22it/s]


    2026-04-08 17:21:59,714 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-08 17:21:59] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:33,  2.70s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:29,  1.86it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.71it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=1024):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=960):  31%|███       | 18/58 [00:03<00:03, 12.07it/s] Compiling num tokens (num_tokens=896):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=832):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]

    Compiling num tokens (num_tokens=768):  31%|███       | 18/58 [00:03<00:03, 12.07it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:03<00:01, 18.17it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:03<00:01, 18.17it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:03<00:01, 18.17it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:03<00:01, 18.17it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:03<00:01, 18.17it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:03<00:01, 18.17it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:03<00:01, 18.17it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:03<00:01, 18.17it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]

    Compiling num tokens (num_tokens=240):  55%|█████▌    | 32/58 [00:03<00:01, 24.85it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:03<00:00, 29.85it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:03<00:00, 29.85it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:03<00:00, 29.85it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:03<00:00, 29.85it/s]Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:03<00:00, 29.85it/s]Compiling num tokens (num_tokens=160):  66%|██████▌   | 38/58 [00:03<00:00, 29.85it/s]Compiling num tokens (num_tokens=144):  66%|██████▌   | 38/58 [00:03<00:00, 29.85it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:03<00:00, 35.12it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:03<00:00, 35.12it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:03<00:00, 35.12it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:03<00:00, 38.40it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:03<00:00, 38.40it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:03<00:00, 38.40it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:03<00:00, 38.40it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:03<00:00, 38.40it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:03<00:00, 38.40it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:03<00:00, 38.40it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:03<00:00, 38.40it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:03<00:00, 38.40it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 15.77it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=120.34 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=120.31 GB):   3%|▎         | 2/58 [00:00<00:03, 17.83it/s]Capturing num tokens (num_tokens=7168 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.83it/s]Capturing num tokens (num_tokens=6656 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   3%|▎         | 2/58 [00:00<00:03, 17.83it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.47it/s]Capturing num tokens (num_tokens=5632 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.47it/s]Capturing num tokens (num_tokens=5120 avail_mem=120.30 GB):   9%|▊         | 5/58 [00:00<00:02, 21.47it/s]Capturing num tokens (num_tokens=4608 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):   9%|▊         | 5/58 [00:00<00:02, 21.47it/s]Capturing num tokens (num_tokens=4096 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3840 avail_mem=120.29 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3584 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3328 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  16%|█▌        | 9/58 [00:00<00:01, 26.24it/s]Capturing num tokens (num_tokens=3072 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.16it/s]Capturing num tokens (num_tokens=2816 avail_mem=120.28 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.16it/s]Capturing num tokens (num_tokens=2560 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.16it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  22%|██▏       | 13/58 [00:00<00:01, 29.16it/s]Capturing num tokens (num_tokens=2304 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.87it/s]Capturing num tokens (num_tokens=2048 avail_mem=120.27 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.87it/s]Capturing num tokens (num_tokens=1792 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.87it/s]Capturing num tokens (num_tokens=1536 avail_mem=120.26 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.87it/s]Capturing num tokens (num_tokens=1280 avail_mem=120.25 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  28%|██▊       | 16/58 [00:00<00:02, 20.87it/s]Capturing num tokens (num_tokens=1024 avail_mem=120.23 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=960 avail_mem=120.25 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.89it/s] Capturing num tokens (num_tokens=896 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=832 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.89it/s]

    Capturing num tokens (num_tokens=768 avail_mem=120.24 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  36%|███▌      | 21/58 [00:00<00:01, 26.89it/s]Capturing num tokens (num_tokens=704 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:01, 31.73it/s]Capturing num tokens (num_tokens=640 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:01, 31.73it/s]Capturing num tokens (num_tokens=576 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:00<00:01, 31.73it/s]Capturing num tokens (num_tokens=512 avail_mem=120.22 GB):  45%|████▍     | 26/58 [00:00<00:01, 31.73it/s]Capturing num tokens (num_tokens=480 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.73it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  45%|████▍     | 26/58 [00:01<00:01, 31.73it/s]Capturing num tokens (num_tokens=448 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.38it/s]Capturing num tokens (num_tokens=416 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.38it/s]Capturing num tokens (num_tokens=384 avail_mem=120.23 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.38it/s]

    Capturing num tokens (num_tokens=352 avail_mem=120.22 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.38it/s]Capturing num tokens (num_tokens=320 avail_mem=120.22 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.38it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  53%|█████▎    | 31/58 [00:01<00:00, 35.38it/s]Capturing num tokens (num_tokens=288 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=256 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=240 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=224 avail_mem=120.21 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=208 avail_mem=120.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  62%|██████▏   | 36/58 [00:01<00:00, 37.95it/s]Capturing num tokens (num_tokens=192 avail_mem=120.20 GB):  71%|███████   | 41/58 [00:01<00:00, 40.04it/s]Capturing num tokens (num_tokens=176 avail_mem=120.20 GB):  71%|███████   | 41/58 [00:01<00:00, 40.04it/s]

    Capturing num tokens (num_tokens=160 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 40.04it/s]Capturing num tokens (num_tokens=144 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 40.04it/s]Capturing num tokens (num_tokens=128 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 40.04it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  71%|███████   | 41/58 [00:01<00:00, 40.04it/s]Capturing num tokens (num_tokens=112 avail_mem=120.19 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.36it/s]Capturing num tokens (num_tokens=96 avail_mem=120.18 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.36it/s] Capturing num tokens (num_tokens=80 avail_mem=120.18 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.36it/s]Capturing num tokens (num_tokens=64 avail_mem=120.18 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.36it/s]Capturing num tokens (num_tokens=48 avail_mem=120.17 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.36it/s]Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  79%|███████▉  | 46/58 [00:01<00:00, 41.36it/s]

    Capturing num tokens (num_tokens=32 avail_mem=120.17 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=28 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=24 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=20 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=16 avail_mem=120.16 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  88%|████████▊ | 51/58 [00:01<00:00, 42.08it/s]Capturing num tokens (num_tokens=12 avail_mem=120.15 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=8 avail_mem=120.15 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.26it/s] Capturing num tokens (num_tokens=4 avail_mem=120.15 GB):  97%|█████████▋| 56/58 [00:01<00:00, 43.26it/s]Capturing num tokens (num_tokens=4 avail_mem=120.15 GB): 100%|██████████| 58/58 [00:01<00:00, 35.05it/s]


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
    Generated text:  Aisha and I am a 17-year-old high school student who has been learning to ride a bike for two months. I have been riding my new bike for the past three days. I have tried on several different brands and models, but it's the one that has the best color and the most interesting design. My favorite bike is a 2018 Honda Revolt 2. I love riding it on our street and on our school campus. My favorite color is red and I enjoy riding it during the summer and autumn. I love to ride it on my bike because it is pretty and has a great sound.
    ===============================
    Prompt: The president of the United States is
    Generated text:  a position of great importance, so he or she is not the most fit person to serve as the leader of the country. The president is a critical position, and it is important for the country to have a president who is fit to serve in this important role. Therefore, in order to select a president who is fit to serve as the leader of the country, a lot of research and analysis must be conducted before the decision is made. The research is done on the characteristics and qualifications of the person who will be serving as the president. It is important to note that there are many factors that can affect the qualifications of a person who will
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. The population of Paris is 2,165,020. Of this number, 48.3% are women. If 16% of the women are married, how many of the women in Paris are married?
    To find the number of women married in Paris, we first need to calculate the number of women in Paris and then determine how many of those women are married. Let's go through the steps:
    
    1. Calculate the number of women in Paris.
       The population of Paris is 2,165,020, and 48.3% of these are
    ===============================
    Prompt: The future of AI is
    Generated text:  bright. In 2022, 52% of jobs in the US will go away due to automation. How does this impact the economy?
    
    Read more: How AI will change jobs and the economy
    
    When it comes to the economy, there’s a lot of speculation about what the future holds. A lot of people are optimistic that the technology will lead to the creation of new, innovative jobs, but others are skeptical of the outlook for traditional jobs as robots take over. In either case, the resulting job market changes are likely to be quite large and have a profound impact on the economy.
    
    The job market is evolving so


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


    Generated text:  [Name] and I'm a [Age] year old [Gender] who has always been fascinated by [Subject]. I'm a [Occupation] who has always been [Motivation] in life. I'm [Personality] and I'm always [Adjective]. I'm [Name] and I'm here to [Purpose]. I'm [Name] and I'm here to [Purpose]. I'm [Name] and I'm here to [Purpose]. I'm [Name] and I'm here to [Purpose]. I'm [Name] and I'm here to [Purpose]. I'm [Name] and I
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as the City of Light, a historic city with a rich history and a vibrant culture. It is located in the south of France and is the largest city in the country. Paris is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, as well as its diverse and multicultural population. The city is also home to many world-renowned museums, theaters, and restaurants, making it a popular tourist destination. Paris is a city of contrasts, with its modern architecture and high-tech industries, as well as its traditional French charm and cultural heritage. Its status as
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased focus on ethical considerations: As AI becomes more prevalent in our daily lives, there will be a greater emphasis on ethical considerations. This will include issues such as bias, transparency, accountability, and the potential for AI to be used for malicious purposes.
    
    2. Integration with other technologies: AI is likely to become more integrated with other technologies, such as blockchain, IoT, and cloud computing. This will allow for more efficient and effective use of AI, as well as the
    


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
    Generated text:  [Name] and I am a [ occupation ]! I have always been fascinated by [x] and my story started when I was [x] years old. I'm a [ x] person who believes in [x] and [x] is the [x] for me. I have always been [x] in my heart and I know that I can achieve anything I set my mind to. I am dedicated to [x] and I will do my best to succeed. I am a [x] of [x] and I am always willing to help others. I love [x] and I know that I am
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris. It is the largest city in France, the second-largest in the European Union, and one of the most populous cities in the world. Paris is home to many world-renowned landmarks, including the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Louvre Museum. The city is known for its rich history, beautiful architecture, and diverse culture, which make it a popular tourist destination. Paris has a vibrant cultural scene and is home to many international institutions and organizations, including the French Embassy and UNESCO. The city is also known for its excellent transportation system, with many modes of public transportation available, including
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  looking more and more like the future of fashion: futuristic and innovative. More and more companies are recognizing the importance of AI in many different industries, from healthcare and finance to manufacturing and transportation. Some of the most exciting developments in AI over the next decade include:
    
    1. Increased use of AI in healthcare: AI can be used to analyze medical images and diagnoses, develop personalized treatment plans, and predict patient outcomes. This will revolutionize how we treat and diagnose diseases, ultimately improving patient outcomes.
    
    2. Improved AI in finance: AI can be used to automate trading strategies, identify fraudulent transactions, and improve customer service. This will enhance financial stability


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

     Alex

    .

     I

    'm

     a

     hard

    working

    ,

     ambitious

     person

     who

     loves

     to

     learn

     new

     things

     and

     stay

     up

     to

     date

     with

     the

     latest

     technology

    .

     I

     have

     a

     sharp

     mind

     and

     a

     passion

     for

     problem

    -solving

    ,

     and

     I

    'm

     always

     looking

     for

     ways

     to

     improve

     my

     skills

     and

     knowledge

    .

     I

    'm

     also

     a

     great

     communicator

     and

     love

     to

     collaborate

     with

     others

    ,

     whether

     it

    's

     through

     teamwork

     or

     through

     online

     discussion

     forums

    .

     I

     enjoy

     traveling

     and

     trying

     new

     foods

    ,

     and

     I

    'm

     always

     eager

     to

     try

     something

     new

    .

     And

     most

     importantly

    ,

     I

    'm

     a

     curious

     person

     who

     always

     wants

     to

     know

     more

     about

     the

     world

     around

     me

    .

     I

    'm

     a

     person

     who

     values

     kindness

     and

     compassion

     and

     am

     always

     willing

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     "

    La

     Ville

     Fl

    uv

    iale

    ,"

     and

     is

     the

     largest

     city

     in

     the

     European

     Union

     and

     the

     world

    's

     largest

     metropolitan

     area

    .

     It

     is

     located

     on

     the

     river

     Se

    ine

     in

     the

     center

     of

     the

     Lo

    ire

     Valley

    ,

     near

     the

     western

     coast

     of

     the

     Mediterranean

     Sea

    .

     Paris

     is

     a

     historic

     center

     of

     France

    ,

     known

     for

     its

     medieval

     architecture

    ,

     art

    ,

     and

     fashion

    ,

     and

     is

     the

     birth

    place

     of

     many

     French

     political

    ,

     artistic

    ,

     and

     scientific

     figures

    .

     It

     is

     also

     the

     world

    's

     most

     important

     cultural

     and

     business

     center

    ,

     and

     hosts

     the

     world

    's

     largest

     opera

     house

    ,

     the

     Op

    éra

     National

     de

     France

    ,

     and

     the

     most

     popular

     music

     festivals

    ,

     such

     as

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     shaped

     by

     a

     combination

     of

     advances

     in

     hardware

    ,

     software

    ,

     and

     algorithms

    ,

     as

     well

     as

     evolving

     societal

     and

     cultural

     norms

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     that

     could

     be

     expected

     to

     shape

     the

     industry

     over

     the

     next

     few

     decades

    :
    


    1

    .

     Increased

     integration

     of

     AI

     into

     everyday

     life

    :

     With

     the

     increasing

     availability

     of

     data

    ,

     AI

     is

     becoming

     more

     accessible

     to

     the

     public

     and

     becoming

     integrated

     into

     everyday

     life

    .

     This

     could

     lead

     to

     a

     more

     connected

    ,

     interconnected

     world

    ,

     where

     people

     are

     increasingly

     relying

     on

     AI

     to

     make

     decisions

     and

     perform

     tasks

    .
    


    2

    .

     Greater

     reliance

     on

     AI

     for

     decision

    -making

    :

     AI

     will

     become

     more

     integrated

     into

     decision

    -making

     processes

    ,

     particularly

     in

     areas

     where

    



```python
llm.shutdown()
```
