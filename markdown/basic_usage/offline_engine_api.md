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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.70it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.70it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:03<03:46,  3.97s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:03<03:46,  3.97s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:03<03:46,  3.97s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<03:46,  3.97s/it]Compiling num tokens (num_tokens=6144):   2%|▏         | 1/58 [00:04<03:46,  3.97s/it]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=5120):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4608):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=4096):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3840):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3584):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]

    Compiling num tokens (num_tokens=3328):   9%|▊         | 5/58 [00:04<00:32,  1.63it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2816):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2560):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2304):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2048):  21%|██        | 12/58 [00:04<00:09,  4.84it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:05,  7.57it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:05,  7.57it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:05,  7.57it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:05,  7.57it/s]

    Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:05,  7.57it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:04<00:05,  7.57it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:04<00:03, 10.75it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:04<00:02, 14.52it/s]

    Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:04<00:02, 14.52it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:04<00:01, 18.54it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:04<00:01, 18.54it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:04<00:01, 18.54it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:04<00:01, 18.54it/s]Compiling num tokens (num_tokens=288):  55%|█████▌    | 32/58 [00:04<00:01, 18.54it/s]Compiling num tokens (num_tokens=256):  55%|█████▌    | 32/58 [00:04<00:01, 18.54it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:04<00:00, 22.86it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:04<00:00, 22.86it/s]

    Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:04<00:00, 22.86it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:04<00:00, 22.86it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:04<00:00, 22.86it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:04<00:00, 22.86it/s]Compiling num tokens (num_tokens=160):  64%|██████▍   | 37/58 [00:04<00:00, 22.86it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:04<00:00, 28.83it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:04<00:00, 28.83it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:04<00:00, 28.83it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:04<00:00, 28.83it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:04<00:00, 28.83it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:04<00:00, 28.83it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:04<00:00, 28.83it/s]

    Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:05<00:00, 33.86it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:05<00:00, 33.86it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:05<00:00, 43.02it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:05<00:00, 43.02it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.33it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=70.78 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.75 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=70.75 GB):   3%|▎         | 2/58 [00:00<00:03, 18.32it/s]Capturing num tokens (num_tokens=7168 avail_mem=70.75 GB):   3%|▎         | 2/58 [00:00<00:03, 18.32it/s]Capturing num tokens (num_tokens=6656 avail_mem=70.74 GB):   3%|▎         | 2/58 [00:00<00:03, 18.32it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=70.74 GB):   7%|▋         | 4/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=6144 avail_mem=70.74 GB):   7%|▋         | 4/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=5632 avail_mem=70.74 GB):   7%|▋         | 4/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.73 GB):   7%|▋         | 4/58 [00:00<00:02, 18.77it/s]Capturing num tokens (num_tokens=5120 avail_mem=70.73 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.74it/s]Capturing num tokens (num_tokens=4608 avail_mem=70.32 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.74it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=70.18 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.74it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.69 GB):  12%|█▏        | 7/58 [00:00<00:02, 22.74it/s]Capturing num tokens (num_tokens=3840 avail_mem=70.69 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.16it/s]Capturing num tokens (num_tokens=3584 avail_mem=70.68 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.16it/s]Capturing num tokens (num_tokens=3328 avail_mem=70.63 GB):  17%|█▋        | 10/58 [00:00<00:03, 14.16it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=70.63 GB):  21%|██        | 12/58 [00:00<00:03, 14.45it/s]Capturing num tokens (num_tokens=3072 avail_mem=70.68 GB):  21%|██        | 12/58 [00:00<00:03, 14.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.27 GB):  21%|██        | 12/58 [00:00<00:03, 14.45it/s]Capturing num tokens (num_tokens=2816 avail_mem=70.27 GB):  24%|██▍       | 14/58 [00:00<00:02, 15.00it/s]Capturing num tokens (num_tokens=2560 avail_mem=70.67 GB):  24%|██▍       | 14/58 [00:00<00:02, 15.00it/s]Capturing num tokens (num_tokens=2304 avail_mem=70.30 GB):  24%|██▍       | 14/58 [00:00<00:02, 15.00it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=70.30 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.66it/s]Capturing num tokens (num_tokens=2048 avail_mem=70.66 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.65 GB):  28%|██▊       | 16/58 [00:01<00:02, 15.66it/s]Capturing num tokens (num_tokens=1792 avail_mem=70.65 GB):  31%|███       | 18/58 [00:01<00:02, 16.67it/s]Capturing num tokens (num_tokens=1536 avail_mem=70.34 GB):  31%|███       | 18/58 [00:01<00:02, 16.67it/s]Capturing num tokens (num_tokens=1280 avail_mem=70.65 GB):  31%|███       | 18/58 [00:01<00:02, 16.67it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=70.65 GB):  34%|███▍      | 20/58 [00:01<00:02, 17.21it/s]Capturing num tokens (num_tokens=1024 avail_mem=70.63 GB):  34%|███▍      | 20/58 [00:01<00:02, 17.21it/s]Capturing num tokens (num_tokens=960 avail_mem=70.39 GB):  34%|███▍      | 20/58 [00:01<00:02, 17.21it/s] Capturing num tokens (num_tokens=896 avail_mem=70.63 GB):  34%|███▍      | 20/58 [00:01<00:02, 17.21it/s]Capturing num tokens (num_tokens=896 avail_mem=70.63 GB):  40%|███▉      | 23/58 [00:01<00:01, 19.22it/s]Capturing num tokens (num_tokens=832 avail_mem=70.63 GB):  40%|███▉      | 23/58 [00:01<00:01, 19.22it/s]Capturing num tokens (num_tokens=768 avail_mem=70.43 GB):  40%|███▉      | 23/58 [00:01<00:01, 19.22it/s]

    Capturing num tokens (num_tokens=704 avail_mem=70.47 GB):  40%|███▉      | 23/58 [00:01<00:01, 19.22it/s]Capturing num tokens (num_tokens=704 avail_mem=70.47 GB):  45%|████▍     | 26/58 [00:01<00:01, 21.13it/s]Capturing num tokens (num_tokens=640 avail_mem=70.60 GB):  45%|████▍     | 26/58 [00:01<00:01, 21.13it/s]Capturing num tokens (num_tokens=576 avail_mem=70.61 GB):  45%|████▍     | 26/58 [00:01<00:01, 21.13it/s]Capturing num tokens (num_tokens=512 avail_mem=70.58 GB):  45%|████▍     | 26/58 [00:01<00:01, 21.13it/s]Capturing num tokens (num_tokens=512 avail_mem=70.58 GB):  50%|█████     | 29/58 [00:01<00:01, 21.81it/s]

    Capturing num tokens (num_tokens=480 avail_mem=70.44 GB):  50%|█████     | 29/58 [00:01<00:01, 21.81it/s]Capturing num tokens (num_tokens=448 avail_mem=70.56 GB):  50%|█████     | 29/58 [00:01<00:01, 21.81it/s]Capturing num tokens (num_tokens=416 avail_mem=70.59 GB):  50%|█████     | 29/58 [00:01<00:01, 21.81it/s]Capturing num tokens (num_tokens=416 avail_mem=70.59 GB):  55%|█████▌    | 32/58 [00:01<00:01, 21.36it/s]Capturing num tokens (num_tokens=384 avail_mem=70.58 GB):  55%|█████▌    | 32/58 [00:01<00:01, 21.36it/s]Capturing num tokens (num_tokens=352 avail_mem=70.57 GB):  55%|█████▌    | 32/58 [00:01<00:01, 21.36it/s]Capturing num tokens (num_tokens=320 avail_mem=70.56 GB):  55%|█████▌    | 32/58 [00:01<00:01, 21.36it/s]

    Capturing num tokens (num_tokens=320 avail_mem=70.56 GB):  60%|██████    | 35/58 [00:01<00:00, 23.29it/s]Capturing num tokens (num_tokens=288 avail_mem=70.55 GB):  60%|██████    | 35/58 [00:01<00:00, 23.29it/s]Capturing num tokens (num_tokens=256 avail_mem=70.52 GB):  60%|██████    | 35/58 [00:01<00:00, 23.29it/s]Capturing num tokens (num_tokens=240 avail_mem=70.52 GB):  60%|██████    | 35/58 [00:01<00:00, 23.29it/s]Capturing num tokens (num_tokens=224 avail_mem=70.53 GB):  60%|██████    | 35/58 [00:01<00:00, 23.29it/s]Capturing num tokens (num_tokens=224 avail_mem=70.53 GB):  67%|██████▋   | 39/58 [00:01<00:00, 26.23it/s]Capturing num tokens (num_tokens=208 avail_mem=70.51 GB):  67%|██████▋   | 39/58 [00:01<00:00, 26.23it/s]Capturing num tokens (num_tokens=192 avail_mem=70.52 GB):  67%|██████▋   | 39/58 [00:01<00:00, 26.23it/s]Capturing num tokens (num_tokens=176 avail_mem=70.52 GB):  67%|██████▋   | 39/58 [00:02<00:00, 26.23it/s]

    Capturing num tokens (num_tokens=160 avail_mem=70.51 GB):  67%|██████▋   | 39/58 [00:02<00:00, 26.23it/s]Capturing num tokens (num_tokens=160 avail_mem=70.51 GB):  74%|███████▍  | 43/58 [00:02<00:00, 29.04it/s]Capturing num tokens (num_tokens=144 avail_mem=70.51 GB):  74%|███████▍  | 43/58 [00:02<00:00, 29.04it/s]Capturing num tokens (num_tokens=128 avail_mem=70.50 GB):  74%|███████▍  | 43/58 [00:02<00:00, 29.04it/s]Capturing num tokens (num_tokens=112 avail_mem=70.49 GB):  74%|███████▍  | 43/58 [00:02<00:00, 29.04it/s]Capturing num tokens (num_tokens=96 avail_mem=70.49 GB):  74%|███████▍  | 43/58 [00:02<00:00, 29.04it/s] Capturing num tokens (num_tokens=96 avail_mem=70.49 GB):  81%|████████  | 47/58 [00:02<00:00, 31.68it/s]Capturing num tokens (num_tokens=80 avail_mem=70.46 GB):  81%|████████  | 47/58 [00:02<00:00, 31.68it/s]Capturing num tokens (num_tokens=64 avail_mem=70.46 GB):  81%|████████  | 47/58 [00:02<00:00, 31.68it/s]Capturing num tokens (num_tokens=48 avail_mem=70.44 GB):  81%|████████  | 47/58 [00:02<00:00, 31.68it/s]

    Capturing num tokens (num_tokens=32 avail_mem=70.46 GB):  81%|████████  | 47/58 [00:02<00:00, 31.68it/s]Capturing num tokens (num_tokens=32 avail_mem=70.46 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.64it/s]Capturing num tokens (num_tokens=28 avail_mem=70.46 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.64it/s]Capturing num tokens (num_tokens=24 avail_mem=70.45 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.64it/s]Capturing num tokens (num_tokens=20 avail_mem=70.45 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.64it/s]Capturing num tokens (num_tokens=16 avail_mem=70.44 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.64it/s]Capturing num tokens (num_tokens=12 avail_mem=70.27 GB):  88%|████████▊ | 51/58 [00:02<00:00, 33.64it/s]Capturing num tokens (num_tokens=12 avail_mem=70.27 GB):  97%|█████████▋| 56/58 [00:02<00:00, 36.04it/s]Capturing num tokens (num_tokens=8 avail_mem=70.27 GB):  97%|█████████▋| 56/58 [00:02<00:00, 36.04it/s] Capturing num tokens (num_tokens=4 avail_mem=70.24 GB):  97%|█████████▋| 56/58 [00:02<00:00, 36.04it/s]Capturing num tokens (num_tokens=4 avail_mem=70.24 GB): 100%|██████████| 58/58 [00:02<00:00, 23.73it/s]


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
    Generated text:  Emma, and I am a student at the University of Waterloo. I have been running a business in the field of mobile technology for the past four years. I have also participated in the Summer Internship program and the RoboCamp program. I am currently pursuing a Master's degree in Computer Science at the University of Waterloo. I have also worked as an intern at one of the big companies in the industry and have been involved in many projects. I have been involved in a variety of projects, from web development to AI research. My previous experience includes working in PHP, Java, and C++. I am a very good communicator and I have
    ===============================
    Prompt: The president of the United States is
    Generated text:  trying to decide whether to use an executive order or a proclamation. An executive order is a type of executive branch action that can be used to set a new standard for the actions of federal government officials. A proclamation is a type of law made by the president in the executive branch. The president of the United States has the authority to issue a proclamation under specific conditions. Here's the schedule of important dates for the proclamation and executive order in the United States government:
    
    - **2023-01-01** - The deadline for making a proclamation
    - **2023-01-15** - The
    ===============================
    Prompt: The capital of France is
    Generated text:  Paris. Paris is the largest city and the most populous (as of 2017) and largest metropolitan (population greater than 50,000) city in the European Union. It is located on the Seine river, in the Marne Valley, in the Loire Valley. The city is situated on the banks of the Seine river, at the confluence of the Dôme and the Marne valleys, and at the confluence of the river Loire with the Seine. The city is situated in the centre of the plain of the Loire Valley.
    
    Based on that paragraph can we conclude that
    ===============================
    Prompt: The future of AI is
    Generated text:  changing the way we work, learn, and even play. The research and development of AI technologies has become crucial to many companies and industries, and the impact on society is profound. AI is evolving in a way that is increasingly disruptive, and it poses a significant threat to job security and employment. The employment sector is currently facing challenges due to the adoption of AI technologies and the rise of automation. However, there is also a growing recognition of the importance of retaining and training workers in the digital age. As such, it is essential to understand the impact of AI on the workforce and how to adapt to these changes. 
    
    To understand the impact


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [age] year old, and I have a [major] degree in [field of study]. I'm a [occupation] with a passion for [interest or hobby]. I'm always looking for new challenges and opportunities to grow and learn. What's your favorite hobby or activity? I love [mention a hobby or activity]. I'm always looking for new ways to challenge myself and expand my knowledge. What's your favorite book or movie?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also home to the French Parliament, the French Academy of Sciences, and the French National Library. Paris is a cultural and economic center with a rich history dating back to the Roman Empire and the Renaissance. It is a popular tourist destination and a major hub for business and commerce in Europe. The city is known for its diverse cuisine, including French cuisine, and its vibrant nightlife. Paris is a city of contrasts, with its modern architecture and historical landmarks blending seamlessly into one another. It is a
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by a number of trends that are expected to shape the technology's direction and impact on society. Here are some of the most likely trends:
    
    1. Increased automation: AI is expected to become more and more integrated into the production process, from manufacturing to healthcare. This will lead to increased automation of tasks, such as manufacturing, transportation, and customer service, which will require more human workers to perform the tasks.
    
    2. AI ethics and privacy: As AI becomes more integrated into our lives, there will be a need for ethical guidelines and regulations to ensure that AI is used in a responsible and fair manner. This will require
    


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
    Generated text:  [Name], and I am a [insert age, height, and weight] year-old female human. I have a [insert hair color] and [insert eye color] who has lived on this planet for the last [insert number of years] years. I have lived a life filled with incredible experiences, from discovering my talents in music to mastering the art of flight and forming bonds with the most incredible creatures. I have a love for learning and always strive to expand my knowledge and understanding of the world around me. I am a [insert personality trait, like a natural leader, kind, adventurous, etc.], and I am always
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, the city of lights, with a population of over 2.7 million people. It is renowned for its rich history and cultural heritage, featuring iconic landmarks such as the Eiffel Tower, Notre Dame Cathedral, and Louvre Museum, as well as its diverse cuisine, fashion, and art scene. The city is also known for its annual cultural festivals, including the Opéra Garnier and the Théâtre du Soleil. Paris is a vibrant and exciting place to visit, with a population of over 2.7 million people and a UNESCO World Heritage site, the Eiffel Tower. It's an essential
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to involve a wide range of trends and developments, including:
    
    1. Increased automation: AI will continue to automate a significant portion of jobs, freeing up more human resources to focus on more complex tasks.
    
    2. Improved privacy and security: As AI becomes more sophisticated, it will require greater privacy and security measures to protect user data.
    
    3. Greater use of machine learning: AI is likely to become more ubiquitous, with more developers and businesses using machine learning to improve efficiency and accuracy.
    
    4. Increased focus on AI ethics: As AI becomes more prevalent, there will be increased focus on ethical considerations and responsible development.
    
    5. More interactions with


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

    Name

    ]

     and

     I

     am

     a

     [

    Age

    ]

     year

     old

     [

    Occup

    ation

    ]

     who

     have

     been

     in

     this

     field

     of

     work

     for

     [

    Number

     of

     Years

    ]

     years

    .

     I

     have

     a

     strong

     work

     ethic

     and

     always

     strive

     to

     exceed

     expectations

     in

     all

     my

     endeavors

    .

     I

     am

     always

     looking

     for

     new

     challenges

     and

     opportunities

     to

     grow

     and

     learn

    .

     I

     am

     a

     detail

    -oriented

     person

     who

     takes

     pride

     in

     my

     work

     and

     always

     strive

     to

     deliver

     the

     best

     results

     possible

    .

     I

     am

     passionate

     about

     my

     profession

     and

     always

     strive

     to

     do

     my

     best

     for

     my

     clients

    .

     I

     am

     dedicated

     and

     persistent

     in

     pursuing

     my

     goals

    ,

     and

     I

     am

     always

     striving

     to

     improve

     and

     become

     even

     more

     proficient

     in

     my

     field

    .

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     also

     known

     as

     the

     City

     of

     Love

    .

     It

     is

     the

     largest

     city

     in

     France

     by

     population

     and

     serves

     as

     the

     seat

     of

     the

     Government

    ,

     the

     State

     and

     most

     of

     the

     French

     society

    .

     Paris

     is

     also

     famous

     for

     its

     iconic

     landmarks

    ,

     including

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

     and

     Notre

     Dame

     Cathedral

    .

     The

     French

     capital

     is

     located

     on

     the

     banks

     of

     the

     Se

    ine

     River

     and

     is

     surrounded

     by

     its

     picturesque

     countryside

     and

     the

     beautiful

     Paris

    ian

     districts

    ,

     including

     Mont

    mart

    re

    ,

     the

     Se

    ine

     River

     Walk

    ,

     Mont

    mart

    re

    ,

     and

     the

     Gothic

     Quarter

    .

     Paris

     is

     also

     known

     for

     its

     food

    ,

     fashion

    ,

     and

     music

    ,

     as

     well

     as

     its

     art

     and

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     promising

     and

     will

     continue

     to

     evolve

     rapidly

    .

     Here

     are

     some

     possible

     trends

     in

     AI

     in

     the

     coming

     years

    :
    


    1

    .

     Increased

     integration

     with

     human

     intelligence

    :

     AI

     will

     continue

     to

     learn

     from

     and

     interact

     with

     humans

    ,

     leading

     to

     more

     intelligent

     and

     adaptable

     systems

    .
    


    2

    .

     Greater

     use

     of

     AI

     for

     critical

     decision

    -making

    :

     AI

     will

     be

     integrated

     into

     business

     and

     decision

    -making

     processes

    ,

     helping

     organizations

     make

     better

    -in

    formed

     decisions

     and

     achieve

     better

     outcomes

    .
    


    3

    .

     Greater

     use

     of

     AI

     for

     healthcare

    :

     AI

     will

     be

     used

     in

     healthcare

     to

     assist

     doctors

     in

     diagn

    osing

     and

     treating

     diseases

    ,

     improving

     patient

     outcomes

     and

     reducing

     medical

     errors

    .
    


    4

    .

     Greater

     use

     of

     AI

     for

     automation

    :

     AI

     will

     be

     used

     to

    



```python
llm.shutdown()
```
