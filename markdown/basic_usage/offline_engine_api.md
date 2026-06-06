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

    Multi-thread loading shards:   0% Completed | 0/1 [00:00<?, ?it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.68it/s]Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  5.67it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=7168):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6656):   2%|▏         | 1/58 [00:04<04:10,  4.40s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=5632):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=5120):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=4608):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]Compiling num tokens (num_tokens=4096):   7%|▋         | 4/58 [00:04<00:46,  1.16it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=3328):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=3072):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2816):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2560):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2304):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2048):  16%|█▌        | 9/58 [00:04<00:15,  3.24it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=1024):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=960):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s] Compiling num tokens (num_tokens=896):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=832):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]

    Compiling num tokens (num_tokens=768):  29%|██▉       | 17/58 [00:04<00:05,  7.62it/s]Compiling num tokens (num_tokens=768):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=704):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=640):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=576):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=512):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=480):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=448):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=416):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=384):  43%|████▎     | 25/58 [00:04<00:02, 13.05it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:04<00:01, 19.36it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:04<00:01, 19.36it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:04<00:01, 19.36it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:04<00:01, 19.36it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:04<00:01, 19.36it/s]

    Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:05<00:01, 19.36it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:05<00:01, 19.36it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:05<00:01, 19.36it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:05<00:00, 25.01it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:05<00:00, 25.01it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:05<00:00, 25.01it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:05<00:00, 25.01it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:05<00:00, 25.01it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:05<00:00, 25.01it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:05<00:00, 25.01it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:05<00:00, 25.01it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:05<00:00, 31.16it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:05<00:00, 31.16it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:05<00:00, 31.16it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:05<00:00, 31.16it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:05<00:00, 31.16it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:05<00:00, 31.16it/s]Compiling num tokens (num_tokens=24):  81%|████████  | 47/58 [00:05<00:00, 31.16it/s]

    Compiling num tokens (num_tokens=20):  81%|████████  | 47/58 [00:05<00:00, 31.16it/s]Compiling num tokens (num_tokens=16):  81%|████████  | 47/58 [00:05<00:00, 31.16it/s]Compiling num tokens (num_tokens=12):  81%|████████  | 47/58 [00:05<00:00, 31.16it/s]Compiling num tokens (num_tokens=8):  81%|████████  | 47/58 [00:05<00:00, 31.16it/s] Compiling num tokens (num_tokens=4):  81%|████████  | 47/58 [00:05<00:00, 31.16it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 44.35it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:05<00:00, 11.02it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=74.24 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.21 GB):   0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=7680 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:04, 11.48it/s]Capturing num tokens (num_tokens=7168 avail_mem=74.21 GB):   3%|▎         | 2/58 [00:00<00:04, 11.48it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=74.20 GB):   3%|▎         | 2/58 [00:00<00:04, 11.48it/s]Capturing num tokens (num_tokens=6656 avail_mem=74.20 GB):   7%|▋         | 4/58 [00:00<00:03, 14.04it/s]Capturing num tokens (num_tokens=6144 avail_mem=74.20 GB):   7%|▋         | 4/58 [00:00<00:03, 14.04it/s]Capturing num tokens (num_tokens=5632 avail_mem=74.20 GB):   7%|▋         | 4/58 [00:00<00:03, 14.04it/s]Capturing num tokens (num_tokens=5120 avail_mem=74.19 GB):   7%|▋         | 4/58 [00:00<00:03, 14.04it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=74.19 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.22it/s]Capturing num tokens (num_tokens=4608 avail_mem=74.18 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.22it/s]Capturing num tokens (num_tokens=4096 avail_mem=74.18 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.22it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.18 GB):  12%|█▏        | 7/58 [00:00<00:02, 17.22it/s]Capturing num tokens (num_tokens=3840 avail_mem=74.18 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.34it/s]Capturing num tokens (num_tokens=3584 avail_mem=74.17 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.34it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  17%|█▋        | 10/58 [00:00<00:02, 19.34it/s]Capturing num tokens (num_tokens=3328 avail_mem=74.17 GB):  21%|██        | 12/58 [00:00<00:02, 18.56it/s]Capturing num tokens (num_tokens=3072 avail_mem=74.17 GB):  21%|██        | 12/58 [00:00<00:02, 18.56it/s]Capturing num tokens (num_tokens=2816 avail_mem=74.17 GB):  21%|██        | 12/58 [00:00<00:02, 18.56it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.16 GB):  21%|██        | 12/58 [00:00<00:02, 18.56it/s]Capturing num tokens (num_tokens=2560 avail_mem=74.16 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.43it/s]Capturing num tokens (num_tokens=2304 avail_mem=74.16 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.43it/s]Capturing num tokens (num_tokens=2048 avail_mem=74.16 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.43it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=74.15 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  26%|██▌       | 15/58 [00:00<00:02, 21.43it/s]Capturing num tokens (num_tokens=1536 avail_mem=74.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=1280 avail_mem=74.15 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=1024 avail_mem=74.13 GB):  33%|███▎      | 19/58 [00:00<00:01, 25.11it/s]Capturing num tokens (num_tokens=960 avail_mem=74.14 GB):  33%|███▎      | 19/58 [00:01<00:01, 25.11it/s] Capturing num tokens (num_tokens=960 avail_mem=74.14 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.37it/s]Capturing num tokens (num_tokens=896 avail_mem=74.14 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.37it/s]

    Capturing num tokens (num_tokens=832 avail_mem=74.14 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.37it/s]Capturing num tokens (num_tokens=768 avail_mem=74.13 GB):  38%|███▊      | 22/58 [00:01<00:01, 24.37it/s]Capturing num tokens (num_tokens=768 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.70it/s]Capturing num tokens (num_tokens=704 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.70it/s]Capturing num tokens (num_tokens=640 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.70it/s]Capturing num tokens (num_tokens=576 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.70it/s]Capturing num tokens (num_tokens=512 avail_mem=74.11 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.70it/s]Capturing num tokens (num_tokens=480 avail_mem=74.13 GB):  43%|████▎     | 25/58 [00:01<00:01, 25.70it/s]Capturing num tokens (num_tokens=480 avail_mem=74.13 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.64it/s]Capturing num tokens (num_tokens=448 avail_mem=74.13 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.64it/s]Capturing num tokens (num_tokens=416 avail_mem=74.12 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.64it/s]Capturing num tokens (num_tokens=384 avail_mem=74.12 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.64it/s]

    Capturing num tokens (num_tokens=352 avail_mem=74.12 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.64it/s]Capturing num tokens (num_tokens=320 avail_mem=74.11 GB):  52%|█████▏    | 30/58 [00:01<00:00, 31.64it/s]Capturing num tokens (num_tokens=320 avail_mem=74.11 GB):  60%|██████    | 35/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=288 avail_mem=74.11 GB):  60%|██████    | 35/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=256 avail_mem=74.11 GB):  60%|██████    | 35/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=240 avail_mem=74.10 GB):  60%|██████    | 35/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=224 avail_mem=74.10 GB):  60%|██████    | 35/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=208 avail_mem=74.09 GB):  60%|██████    | 35/58 [00:01<00:00, 36.24it/s]Capturing num tokens (num_tokens=208 avail_mem=74.09 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=192 avail_mem=74.09 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=176 avail_mem=74.09 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=160 avail_mem=74.09 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.72it/s]

    Capturing num tokens (num_tokens=144 avail_mem=74.08 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  69%|██████▉   | 40/58 [00:01<00:00, 39.72it/s]Capturing num tokens (num_tokens=128 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=112 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=96 avail_mem=74.08 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.10it/s] Capturing num tokens (num_tokens=80 avail_mem=74.07 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=64 avail_mem=74.07 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=48 avail_mem=74.06 GB):  78%|███████▊  | 45/58 [00:01<00:00, 42.10it/s]Capturing num tokens (num_tokens=48 avail_mem=74.06 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.38it/s]Capturing num tokens (num_tokens=32 avail_mem=74.06 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.38it/s]Capturing num tokens (num_tokens=28 avail_mem=74.06 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.38it/s]Capturing num tokens (num_tokens=24 avail_mem=74.05 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.38it/s]

    Capturing num tokens (num_tokens=20 avail_mem=74.05 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.38it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  86%|████████▌ | 50/58 [00:01<00:00, 43.38it/s]Capturing num tokens (num_tokens=16 avail_mem=74.05 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=12 avail_mem=74.04 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=8 avail_mem=74.04 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.68it/s] Capturing num tokens (num_tokens=4 avail_mem=74.04 GB):  95%|█████████▍| 55/58 [00:01<00:00, 44.68it/s]Capturing num tokens (num_tokens=4 avail_mem=74.04 GB): 100%|██████████| 58/58 [00:01<00:00, 31.41it/s]


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
    Generated text:  Elisa DeMarzo and I am currently working as a partner at The American Academy of Pediatrics. My goal is to advocate for the role of the child as the primary caregiver and advocate for the best interests of the child. My career began as a school nurse and has since expanded into maternal-fetal medicine. I specialize in the diagnosis and treatment of fetal and neonatal complications, including birth trauma, neonatal care, and treatment of infantile respiratory distress syndrome. My areas of expertise also include neonatal intensive care unit (NICU), neonatal care, and neonatal resuscitation. I have taught in the United States, Canada
    ===============================
    Prompt: The president of the United States is
    Generated text:  the head of the executive branch and serves as the commander-in-chief of the armed forces. Which of the following statements is incorrect? A. The president is a representative of the people. B. The president is a citizen. C. The president has the power of appointment. D. The president is the commander-in-chief of the armed forces. Which of the following statements is incorrect?
    Answer:
    B
    
    Which of the following statements about the use of physiological saline is correct?
    A. It has no medical value
    B. It should be diluted with normal saline
    C. It can be used as a substitute for oral intake of other solutions
    ===============================
    Prompt: The capital of France is
    Generated text:  a city with a population of 675,000 and the largest. The main character of the novel is named after its name. The name of the capital is Paris. What is the capital city? Paris is the capital city of France. It has a population of approximately 675,000 and is the largest city in the country. The main character in the novel is named after Paris. There is no specific information given about the name of the character, but it is likely to be named after Paris in the novel. However, without more context, it is not possible to determine the exact name of
    ===============================
    Prompt: The future of AI is
    Generated text:  an ongoing conversation that will likely continue for years to come, but one thing is clear: the applications of AI are growing and will likely continue to grow as the technology continues to improve and evolve. In this article, we will explore the potential applications of AI and how it will continue to evolve in the future.
    1. Healthcare: AI is already being used in healthcare to improve patient outcomes and reduce costs. For example, AI algorithms can analyze medical images and diagnoses to help doctors make more accurate diagnoses. In the future, AI could be used to identify potential health risks before they become serious, and to develop personalized treatment plans for patients.
    2


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


    Generated text:  [Name], and I'm a [job title] at [company name]. I'm excited to meet you and learn more about your career. What can you tell me about yourself? I'm a [insert a unique skill or personality trait here]. I enjoy [insert a hobby or activity here]. What's your favorite hobby or activity? I'm always looking for new experiences and challenges to try. What's your favorite hobby or activity? I'm always looking for new experiences and challenges to try. What's your favorite hobby or activity? I'm always looking for new experiences and challenges to try. What's your favorite hobby or activity?
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is


    Generated text:  Paris, the city known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. It is also a major cultural and economic center, hosting numerous museums, theaters, and other cultural institutions. Paris is known for its rich history, including the influence of the French Revolution and the influence of the French Revolution on art, literature, and philosophy. It is also home to the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral. Paris is a popular tourist destination and a major economic hub in France. The city is also known for its cuisine, including French cuisine
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is


    Generated text:  likely to be characterized by several key trends:
    
    1. Increased integration with human intelligence: AI systems will become more integrated with human intelligence, allowing them to learn from and adapt to human behavior and decision-making processes. This will enable more sophisticated and adaptive AI systems that can handle complex and unpredictable situations.
    
    2. Enhanced privacy and security: As AI systems become more integrated with human intelligence, there will be increased concerns about privacy and security. There will be a need for more robust privacy and security measures to protect the data and information that AI systems collect and process.
    
    3. Greater automation and efficiency: AI systems will become more efficient and effective at performing
    


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
    Generated text:  [Name] and I am a/an [occupation] with [number] years of experience in [industry/field]. I am currently working in the [industry/field] as a/an [occupation] with [number] years of experience, and I have a passion for [the reason why you like this profession]. What are some of the best things you have done in your career so far? What are your goals for the future? What are some of the most challenging parts of your job, and how do you handle them? What are some of the things that are important to you about your job? What are some things that are
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text:  Paris, a bustling and culturally rich city known for its rich history, art, and architecture. It is a UNESCO World Heritage site and a popular tourist destination, with attractions like the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Paris is also famous for its fashion industry, with designers such as Coco Chanel and Yves Saint Laurent. It's a cultural melting pot of different traditions, languages, and cuisines, and the city continues to evolve and innovate, with a strong focus on sustainability and innovation in the future. Paris is the second-largest city in France by population and one of the most cosmopolitan in
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text:  expected to be characterized by several key trends, including:
    
    1. Increased integration with other technologies: AI is increasingly being integrated with other technologies such as machine learning, natural language processing, and computer vision, creating more complex and integrated systems.
    
    2. Enhanced accuracy and precision: AI is being trained on larger and more diverse datasets, which means that AI systems are becoming more accurate and precise at performing tasks such as image and speech recognition, fraud detection, and customer service.
    
    3. Greater transparency and explainability: With the rise of AI systems, there is a greater demand for transparency and explainability in how they work, as well as how they


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

    ],

     and

     I

    'm

     [

    Age

    ].

     I

    'm

     a

     [

    Occup

    ation

     or

     Achievement

    ]

     in

     [

    Occup

    ation

     or

     Achievement

    ].

     I

    'm

     fluent

     in

     [

    Language

     or

     Currency

    ]

     and

     have

     been

     living

     and

     working

     in

     [

    Country

    /

    Region

    ]

     for

     [

    Years

    ].

     I

    'm

     a

     [

    Person

    ality

     or

     Character

    istic

    ]

     who

     loves

     [

    Your

     Interest

     or

     Hobby

    ].

     I

    'm

     a

     [

    Alignment

     or

     Character

     Value

    ]

     who

     is

     always

     [

    Something

    ],

     and

     I

    'm

     [

    Something

    ].

     I

    'm

     a

     [

    Relationship

     Type

    ]

     who

     can

     [

    Something

    ].

     I

    'm

     [

    Something

    ]

     with

     [

    Something

    ]

     who

     is

     [

    Something

    ].

     I

    'm

     a

     [

    Something

    ]

     who

     always

     [

    Something

    ].

     I

    'm

    
    
    Prompt: Provide a concise factual statement about France’s capital city. The capital of France is
    Generated text: 

     Paris

    ,

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

     and

     the

     Notre

    -D

    ame

     Cathedral

    .

     France

    's

     capital

     is

     also

     the

     seat

     of

     the

     French

     government

     and

     home

     to

     the

     country

    's

     cultural

    ,

     educational

    ,

     and

     political

     centers

    .

     The

     city

    's

     rich

     history

     and

     cultural

     heritage

     are

     also

     evident

     in

     its

     numerous

     museums

    ,

     theaters

    ,

     and

     other

     attractions

    .

     Paris

     is

     a

     major

     tourist

     destination

     and

     a

     popular

     destination

     for

     international

     travelers

    .

     Despite

     facing

     some

     challenges

    ,

     Paris

     remains

     a

     vibrant

     and

     vibrant

     city

     with

     a

     rich

     history

     and

     culture

    .

     
    


    Paris

     is

     the

     capital

     of

     France

     and

     is

     the

     home

     to

     the

     country

    's

     cultural

    ,

     educational

    ,

     and

     political

    
    
    Prompt: Explain possible future trends in artificial intelligence. The future of AI is
    Generated text: 

     expected

     to

     continue

     to

     be

     shaped

     by

     a

     variety

     of

     factors

    ,

     including

     advancements

     in

     data

     and

     machine

     learning

    ,

     new

     types

     of

     hardware

     and

     software

    ,

     and

     the

     increasing

     importance

     of

     ethical

     considerations

    .

     Here

     are

     some

     possible

     future

     trends

     in

     AI

    :
    


    1

    .

     Increased

     focus

     on

     ethical

     considerations

    :

     As

     AI

     becomes

     more

     advanced

     and

     widely

     used

    ,

     there

     will

     be

     greater

     emphasis

     on

     ensuring

     that

     it

     is

     used

     eth

    ically

     and

     responsibly

    .

     This

     could

     involve

     developing

     frameworks

     for

     evaluating

     the

     potential

     impact

     of

     AI

     on

     individuals

     and

     society

    ,

     and

     ensuring

     that

     AI

     is

     used

     for

     good

    .
    


    2

    .

     More

     specialized

     AI

    :

     There

     will

     be

     an

     increased

     demand

     for

     AI

     that

     is

     tailored

     to

     specific

     tasks

     and

     applications

    .

     This

     could

    



```python
llm.shutdown()
```
