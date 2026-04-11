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


    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.04it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.03it/s]


    2026-04-11 05:00:11,448 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-11 05:00:11] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:02<02:29,  2.62s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]Compiling num tokens (num_tokens=3840):   7%|▋         | 4/58 [00:02<00:28,  1.91it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=3328):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=3072):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=2816):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=2560):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=2304):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=2048):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=1792):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=1536):  17%|█▋        | 10/58 [00:02<00:08,  5.85it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:02<00:02, 13.15it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:02<00:02, 13.15it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:02<00:02, 13.15it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:02<00:02, 13.15it/s] Compiling num tokens (num_tokens=896):  33%|███▎      | 19/58 [00:02<00:02, 13.15it/s]Compiling num tokens (num_tokens=832):  33%|███▎      | 19/58 [00:03<00:02, 13.15it/s]Compiling num tokens (num_tokens=768):  33%|███▎      | 19/58 [00:03<00:02, 13.15it/s]Compiling num tokens (num_tokens=704):  33%|███▎      | 19/58 [00:03<00:02, 13.15it/s]

    Compiling num tokens (num_tokens=640):  33%|███▎      | 19/58 [00:03<00:02, 13.15it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=352):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=320):  47%|████▋     | 27/58 [00:03<00:01, 20.25it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:03<00:00, 27.96it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:03<00:00, 27.96it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:03<00:00, 27.96it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:03<00:00, 27.96it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:03<00:00, 27.96it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:03<00:00, 27.96it/s]

    Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:03<00:00, 27.96it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:03<00:00, 27.96it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:03<00:00, 34.39it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:03<00:00, 34.39it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:03<00:00, 34.39it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:03<00:00, 34.39it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:03<00:00, 34.39it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:03<00:00, 34.39it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:03<00:00, 34.39it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:03<00:00, 34.39it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:03<00:00, 40.37it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:03<00:00, 40.37it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:03<00:00, 40.37it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:03<00:00, 40.37it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:03<00:00, 40.37it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:03<00:00, 40.37it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:03<00:00, 40.37it/s]

    Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:03<00:00, 40.37it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:03<00:00, 40.37it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:03<00:00, 40.37it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:03<00:00, 16.67it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=76.12 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.09 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=76.09 GB):   3%|▎         | 2/58 [00:00<00:04, 13.39it/s]Capturing num tokens (num_tokens=7168 avail_mem=76.09 GB):   3%|▎         | 2/58 [00:00<00:04, 13.39it/s]Capturing num tokens (num_tokens=6656 avail_mem=76.08 GB):   3%|▎         | 2/58 [00:00<00:04, 13.39it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=76.08 GB):   7%|▋         | 4/58 [00:00<00:03, 14.46it/s]Capturing num tokens (num_tokens=6144 avail_mem=76.09 GB):   7%|▋         | 4/58 [00:00<00:03, 14.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.08 GB):   7%|▋         | 4/58 [00:00<00:03, 14.46it/s]Capturing num tokens (num_tokens=5632 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:03, 15.26it/s]Capturing num tokens (num_tokens=5120 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:03, 15.26it/s]Capturing num tokens (num_tokens=4608 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:03, 15.26it/s]Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  10%|█         | 6/58 [00:00<00:03, 15.26it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=3840 avail_mem=76.08 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=3584 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=3328 avail_mem=76.07 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.06 GB):  16%|█▌        | 9/58 [00:00<00:02, 18.55it/s]Capturing num tokens (num_tokens=3072 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.77it/s]Capturing num tokens (num_tokens=2816 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.77it/s]Capturing num tokens (num_tokens=2560 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.77it/s]Capturing num tokens (num_tokens=2304 avail_mem=76.06 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.77it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  22%|██▏       | 13/58 [00:00<00:01, 22.77it/s]Capturing num tokens (num_tokens=2048 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.73it/s]Capturing num tokens (num_tokens=1792 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.73it/s]Capturing num tokens (num_tokens=1536 avail_mem=76.05 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  29%|██▉       | 17/58 [00:00<00:01, 24.73it/s]Capturing num tokens (num_tokens=1280 avail_mem=76.04 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.33it/s]Capturing num tokens (num_tokens=1024 avail_mem=76.02 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.33it/s]Capturing num tokens (num_tokens=960 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.33it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.33it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  34%|███▍      | 20/58 [00:00<00:01, 24.33it/s]Capturing num tokens (num_tokens=832 avail_mem=76.03 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.17it/s]Capturing num tokens (num_tokens=768 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.17it/s]Capturing num tokens (num_tokens=704 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.17it/s]Capturing num tokens (num_tokens=640 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.17it/s]Capturing num tokens (num_tokens=576 avail_mem=76.02 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.17it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  41%|████▏     | 24/58 [00:01<00:01, 27.17it/s]Capturing num tokens (num_tokens=512 avail_mem=76.01 GB):  50%|█████     | 29/58 [00:01<00:00, 33.01it/s]Capturing num tokens (num_tokens=480 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:01<00:00, 33.01it/s]Capturing num tokens (num_tokens=448 avail_mem=76.02 GB):  50%|█████     | 29/58 [00:01<00:00, 33.01it/s]

    Capturing num tokens (num_tokens=416 avail_mem=75.29 GB):  50%|█████     | 29/58 [00:01<00:00, 33.01it/s]Capturing num tokens (num_tokens=384 avail_mem=74.91 GB):  50%|█████     | 29/58 [00:01<00:00, 33.01it/s]Capturing num tokens (num_tokens=352 avail_mem=74.91 GB):  50%|█████     | 29/58 [00:01<00:00, 33.01it/s]Capturing num tokens (num_tokens=320 avail_mem=74.91 GB):  50%|█████     | 29/58 [00:01<00:00, 33.01it/s]Capturing num tokens (num_tokens=320 avail_mem=74.91 GB):  60%|██████    | 35/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=288 avail_mem=74.90 GB):  60%|██████    | 35/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=256 avail_mem=74.90 GB):  60%|██████    | 35/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=240 avail_mem=74.90 GB):  60%|██████    | 35/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=224 avail_mem=74.89 GB):  60%|██████    | 35/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=208 avail_mem=74.89 GB):  60%|██████    | 35/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=192 avail_mem=74.89 GB):  60%|██████    | 35/58 [00:01<00:00, 38.55it/s]Capturing num tokens (num_tokens=192 avail_mem=74.89 GB):  71%|███████   | 41/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=176 avail_mem=74.89 GB):  71%|███████   | 41/58 [00:01<00:00, 42.66it/s]

    Capturing num tokens (num_tokens=160 avail_mem=74.88 GB):  71%|███████   | 41/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=144 avail_mem=74.88 GB):  71%|███████   | 41/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=128 avail_mem=74.88 GB):  71%|███████   | 41/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=112 avail_mem=74.87 GB):  71%|███████   | 41/58 [00:01<00:00, 42.66it/s]Capturing num tokens (num_tokens=112 avail_mem=74.87 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=96 avail_mem=74.87 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.52it/s] Capturing num tokens (num_tokens=80 avail_mem=74.87 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=64 avail_mem=74.86 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=48 avail_mem=74.86 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=32 avail_mem=74.86 GB):  79%|███████▉  | 46/58 [00:01<00:00, 44.52it/s]Capturing num tokens (num_tokens=32 avail_mem=74.86 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=28 avail_mem=74.85 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.70it/s]

    Capturing num tokens (num_tokens=24 avail_mem=74.85 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=20 avail_mem=74.84 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=16 avail_mem=74.84 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=12 avail_mem=74.84 GB):  88%|████████▊ | 51/58 [00:01<00:00, 45.70it/s]Capturing num tokens (num_tokens=12 avail_mem=74.84 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=8 avail_mem=74.84 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.81it/s] Capturing num tokens (num_tokens=4 avail_mem=74.83 GB):  97%|█████████▋| 56/58 [00:01<00:00, 46.81it/s]Capturing num tokens (num_tokens=4 avail_mem=74.83 GB): 100%|██████████| 58/58 [00:01<00:00, 33.52it/s]


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
    Generated text:  Lukas and I'm a computer science student from Bergen, Norway. I'm passionate about technology and I'm constantly exploring the latest advancements in AI. I'm currently working on developing a natural language processing (NLP) model that can understand and generate natural language using computer code. Can you tell me more about your work in developing NLP models and how they are changing the way we communicate and interact with technology?
    
    As a computer science student, I am always looking for new ways to apply technology to solve real-world problems. One of the most exciting advancements in NLP is the development of language models that can understand and generate human language,
    ===============================
    Prompt: The president of the United States is
    Generated text:  a male. Does this mean that the president of the United States is not a woman?
    No, the president of the United States being a male does not imply that he is not a woman. Both males and females can hold the position of president, and the president of the United States is indeed a male. The title "president" is a political office, and men and women are both considered citizens and capable of holding such positions. The gender of the president does not affect the gender of any other official or political position. Therefore, the statement that the president of the United States is a male does not negate the fact that he is
    ===============================
    Prompt: The capital of France is
    Generated text: : A: Paris B: London C: New York D: Berlin
    The capital of France is A: Paris. Paris is the capital city of France and is known for its iconic landmarks, such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Other cities mentioned in the question are London, New York, and Berlin. London is the capital of England, New York is the capital of the United States, and Berlin is the capital of Germany. There is no other capital city in these options that is the capital of France. The answer to this question is A: Paris.
    ===============================
    Prompt: The future of AI is
    Generated text:  revolutionary, with more and more AI applications being integrated into our daily lives. However, how to ensure the security and privacy of such applications is a serious issue that needs to be addressed. In this article, we will explore some of the most effective ways to ensure the security and privacy of AI applications.
    1. Implementing Data Privacy and Security Measures: One of the most important ways to ensure the security and privacy of AI applications is to implement data privacy and security measures. This includes measures to protect the data, such as encryption, access controls, and regular audits.
    2. Using Encryption: Encryption is an effective way to protect data in transit


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about you. What can you tell me about yourself? I'm a [insert a short description of your character here, such as "a friendly and outgoing person" or "a dedicated and hardworking professional"]. I enjoy [insert a short description of your character trait or hobby here, such as "reading books" or "traveling the world"]. I'm always looking for new experiences and opportunities to learn and grow. What do you like to do in your free time? I enjoy [insert a short description of
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, also known as "La Ville-Marie" or "La Ville de Paris". It is the largest city in France and the second-largest city in the European Union, with a population of over 2. 5 million people. Paris is known for its rich history, art, and culture, and is a major tourist destination. The city is home to many famous landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. Paris is also a major center for business, finance, and education, and is a popular destination for international visitors. The city is also
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI is likely to become more integrated with human intelligence, allowing machines to learn and adapt to human behavior and preferences. This could lead to more personalized and adaptive AI systems.
    
    2. Greater emphasis on ethical considerations: As AI becomes more integrated with human intelligence, there will be a greater emphasis on ethical considerations. This could lead to more rigorous testing and regulation of AI systems to ensure they are safe and ethical.
    
    3. Increased use of AI in healthcare: AI is already being used in healthcare to improve diagnosis, treatment, and patient care. As AI becomes more
    


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
    Generated text:  [Name], and I'm here to help. I'm a [short, neutral description of your profession or role]. In my free time, I enjoy [specific hobby or activity]. If you have any questions or need assistance with anything, please feel free to reach out. How can I assist you today? [Your profession or role, if applicable. Write in a neutral tone to avoid bias]. How can I assist you today?
    Hello, my name is [Name], and I'm here to help. I'm a [short, neutral description of your profession or role]. In my free time, I enjoy [specific hobby or activity
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris.
    Paris is the largest city in France by population and has a rich history and cultural heritage. It is also one of the world’s most famous and popular cities, known for its architecture, museums, and world-renowned cuisine. The city is home to many iconic landmarks, including the Eiffel Tower and the Louvre Museum. Paris is also a major financial center and has a thriving arts scene, with numerous museums, theaters, and galleries. The city is known for its diverse population and cultural diversity, with many ethnic groups living together in the capital. Overall, Paris is a beautiful and dynamic city that is beloved by millions of
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  vast and unpredictable, and it is difficult to predict exactly what will happen. However, here are some possible trends that are likely to influence the AI landscape in the next few years:
    
    1. Increased development of advanced AI techniques: As AI systems become more complex, they will need to be developed to handle more complex tasks. This will lead to the development of new AI techniques such as neural networks, deep learning, and reinforcement learning.
    
    2. Integration of AI with other technologies: AI will likely be integrated with other technologies such as blockchain, quantum computing, and biotechnology. This integration will enable new applications of AI to be realized, such as


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

    role

    /

    position

    ]

     consultant

     in

     the

     industry

    .

     [

    Name

    ]

     is

     my

     go

    -to

     resource

     for

     creating

     and

     refining

     proposal

     documents

    ,

     and

     my

     expertise

     spans

     a

     variety

     of

     industries

    ,

     including

     technology

    ,

     healthcare

    ,

     and

     finance

    .

     [

    Name

    ]

     provides

     client

    -focused

     advice

     to

     help

     clients

     craft

     compelling

     and

     persuasive

     proposals

     that

     stand

     out

     in

     a

     crowded

     market

    .

     My

     goal

     is

     to

     help

     businesses

     like

     yours

     achieve

     their

     goals

    ,

     whether

     it

    's

     by

     providing

     you

     with

     valuable

     insights

    ,

     offering

     solutions

    ,

     or

     simply

     by

     being

     a

     trusted

     advisor

    .

     I

     believe

     that

     every

     proposal

    ,

     no

     matter

     how

     unique

     or

     unconventional

    ,

     has

     the

     power

     to

     make

     a

     significant

     impact

    .

     With

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

     which

     is

     known

     for

     its

     beautiful

     architecture

    ,

     art

    ,

     and

     cultural

     landmarks

    .

     It

     is

     also

     one

     of

     the

     most

     populous

     cities

     in

     the

     world

    ,

     with

     over

     

    2

     million

     residents

    .

     The

     city

     is

     home

     to

     many

     important

     cultural

     institutions

    ,

     including

     the

     Lou

    vre

     Museum

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

     Paris

     is

     a

     popular

     tourist

     destination

     and

     a

     major

     international

     financial

     hub

    .

     Its

     diverse

     neighborhoods

     and

     historic

     landmarks

     make

     it

     a

     charming

     and

     vibrant

     city

     that

     is

     worth

     visiting

    .

     It

     is

     also

     home

     to

     many

     notable

     French

     artists

     and

     writers

    .

     The

     city

     has

     been

     influenced

     by

     various

     European

     cultures

     throughout

     its

     history

    ,

     and

     it

     continues

     to

     play

     an

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     likely

     to

     be

     a

     combination

     of

     many

     different

     trends

    ,

     including

    :
    


    1

    .

     The

     continued

     development

     of

     more

     powerful

     AI

     models

     and

     algorithms

    
    


    2

    .

     The

     increasing

     use

     of

     AI

     in

     fields

     such

     as

     healthcare

    ,

     finance

    ,

     and

     transportation

    
    


    3

    .

     The

     integration

     of

     AI

     into

     everyday

     objects

     and

     devices

    
    


    4

    .

     The

     growing

     importance

     of

     AI

     in

     decision

    -making

     and

     automation

    
    


    5

    .

     The

     increasing

     integration

     of

     AI

     with

     human

     workers

     and

     the

     need

     for

     them

     to

     be

     trained

     to

     work

     with

     AI

    
    


    6

    .

     The

     increasing

     importance

     of

     AI

     in

     creating

     new

     industries

     and

     industries

     of

     the

     future

    
    


    7

    .

     The

     growing

     use

     of

     AI

     in

     creating

     and

     implementing

     policies

     and

     regulations

    
    


    8

    .

     The

     increasing

     use

    



```python
llm.shutdown()
```
