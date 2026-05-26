# Reasoning Parser

SGLang supports parsing reasoning content out from "normal" content for reasoning models such as [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1).

## Supported Models & Parsers

| Model  |  Reasoning tags      | Parser | Notes |
|---------|-----------------------------|------------------|-------|
| [DeepSeek‑R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) | `<think>` … `</think>` | `deepseek-r1` | Supports all variants (R1, R1-0528, R1-Distill) |
| [DeepSeek‑V3 series](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) | `<think>` … `</think>` | `deepseek-v3` | Including [DeepSeek‑V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp). Supports `thinking` parameter |
| [Standard Qwen3 models](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) | `<think>` … `</think>` | `qwen3` | Supports `enable_thinking` parameter |
| [Qwen3-Thinking models](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) | `<think>` … `</think>` | `qwen3` or `qwen3-thinking` | Always generates thinking content |
| [Kimi K2 Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) | `◁think▷` … `◁/think▷` | `kimi_k2` | Uses special thinking delimiters. Also requires `--tool-call-parser kimi_k2` for tool use. |
| [GPT OSS](https://huggingface.co/openai/gpt-oss-120b) | `<\|channel\|>analysis<\|message\|>` … `<\|end\|>` | `gpt-oss` | N/A |
### Model-Specific Behaviors

**DeepSeek-R1 Family:**
- DeepSeek-R1: No `<think>` start tag, jumps directly to thinking content
- DeepSeek-R1-0528: Generates both `<think>` start and `</think>` end tags
- Both are handled by the same `deepseek-r1` parser

**DeepSeek-V3 Family:**
- DeepSeek-V3.1/V3.2: Hybrid model supporting both thinking and non-thinking modes, use the `deepseek-v3` parser and `thinking` parameter (NOTE: not `enable_thinking`)

**Qwen3 Family:**
- Standard Qwen3 (e.g., Qwen3-2507): Use `qwen3` parser, supports `enable_thinking` in chat templates
- Qwen3-Thinking (e.g., Qwen3-235B-A22B-Thinking-2507): Use `qwen3` or `qwen3-thinking` parser, always thinks

**Kimi K2:**
- Kimi K2 Thinking: Uses special `◁think▷` and `◁/think▷` tags. For agentic tool use, also specify `--tool-call-parser kimi_k2`.

**GPT OSS:**
- GPT OSS: Uses special `<|channel|>analysis<|message|>` and `<|end|>` tags

## Usage

### Launching the Server

Specify the `--reasoning-parser` option.


```python
import requests
from openai import OpenAI
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --host 0.0.0.0 --reasoning-parser deepseek-r1 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.40s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.35s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.36s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:08,  5.41s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:08,  5.41s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:18,  2.47s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:18,  2.47s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:24,  1.54s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:24,  1.54s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:58,  1.09s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:58,  1.09s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:07<00:43,  1.21it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:07<00:43,  1.21it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:34,  1.50it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:34,  1.50it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:27,  1.83it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:27,  1.83it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:22,  2.18it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:22,  2.18it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:08<00:18,  2.59it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:08<00:18,  2.59it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:08<00:15,  3.03it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:08<00:15,  3.03it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:13,  3.39it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:13,  3.39it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:11,  3.83it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:11,  3.83it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:10,  4.30it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:10,  4.30it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:09<00:09,  4.81it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:09<00:09,  4.81it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:09<00:08,  5.26it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:09<00:08,  5.26it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:09<00:07,  5.88it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:09<00:07,  5.88it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:06,  6.62it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:06,  6.62it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:05,  7.21it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:05,  7.21it/s]Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:09<00:05,  7.21it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:04,  8.65it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:04,  8.65it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:04,  8.65it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:09<00:03, 10.52it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:09<00:03, 10.52it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:09<00:03, 10.52it/s]

    Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:09<00:02, 12.36it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:09<00:02, 12.36it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:10<00:02, 12.36it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:10<00:02, 12.36it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:10<00:02, 14.96it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:10<00:02, 14.96it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:10<00:02, 14.96it/s]

    Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:10<00:02, 14.96it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:10<00:01, 18.36it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:10<00:01, 18.36it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:10<00:01, 18.36it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:10<00:01, 18.36it/s]Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:10<00:01, 19.69it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:10<00:01, 19.69it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:10<00:01, 19.69it/s]

    Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:10<00:01, 19.69it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:10<00:01, 19.69it/s]Compiling num tokens (num_tokens=256):  64%|██████▍   | 37/58 [00:10<00:00, 23.23it/s]Compiling num tokens (num_tokens=240):  64%|██████▍   | 37/58 [00:10<00:00, 23.23it/s]Compiling num tokens (num_tokens=224):  64%|██████▍   | 37/58 [00:10<00:00, 23.23it/s]Compiling num tokens (num_tokens=208):  64%|██████▍   | 37/58 [00:10<00:00, 23.23it/s]Compiling num tokens (num_tokens=192):  64%|██████▍   | 37/58 [00:10<00:00, 23.23it/s]Compiling num tokens (num_tokens=176):  64%|██████▍   | 37/58 [00:10<00:00, 23.23it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:10<00:00, 28.45it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:10<00:00, 28.45it/s]

    Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:10<00:00, 28.45it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:10<00:00, 28.45it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:10<00:00, 28.45it/s]Compiling num tokens (num_tokens=112):  79%|███████▉  | 46/58 [00:10<00:00, 30.35it/s]Compiling num tokens (num_tokens=96):  79%|███████▉  | 46/58 [00:10<00:00, 30.35it/s] Compiling num tokens (num_tokens=80):  79%|███████▉  | 46/58 [00:10<00:00, 30.35it/s]Compiling num tokens (num_tokens=64):  79%|███████▉  | 46/58 [00:10<00:00, 30.35it/s]Compiling num tokens (num_tokens=48):  79%|███████▉  | 46/58 [00:10<00:00, 30.35it/s]Compiling num tokens (num_tokens=32):  79%|███████▉  | 46/58 [00:10<00:00, 30.35it/s]

    Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:10<00:00, 33.47it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:10<00:00, 33.47it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:10<00:00, 33.47it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:10<00:00, 33.47it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:10<00:00, 33.47it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:10<00:00, 33.47it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:10<00:00, 33.47it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:10<00:00, 39.17it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:10<00:00, 39.17it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=23.14 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=23.14 GB):   2%|▏         | 1/58 [00:00<00:30,  1.89it/s]Capturing num tokens (num_tokens=7680 avail_mem=22.14 GB):   2%|▏         | 1/58 [00:00<00:30,  1.89it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=22.14 GB):   3%|▎         | 2/58 [00:01<00:36,  1.55it/s]Capturing num tokens (num_tokens=7168 avail_mem=22.26 GB):   3%|▎         | 2/58 [00:01<00:36,  1.55it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=22.26 GB):   5%|▌         | 3/58 [00:01<00:32,  1.71it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.19 GB):   5%|▌         | 3/58 [00:01<00:32,  1.71it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.19 GB):   7%|▋         | 4/58 [00:02<00:27,  1.93it/s]Capturing num tokens (num_tokens=6144 avail_mem=41.42 GB):   7%|▋         | 4/58 [00:02<00:27,  1.93it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=41.42 GB):   9%|▊         | 5/58 [00:02<00:25,  2.06it/s]Capturing num tokens (num_tokens=5632 avail_mem=41.48 GB):   9%|▊         | 5/58 [00:02<00:25,  2.06it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=41.48 GB):  10%|█         | 6/58 [00:02<00:23,  2.23it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.18 GB):  10%|█         | 6/58 [00:02<00:23,  2.23it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.18 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.38it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.63 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.38it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=41.63 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.63it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.58 GB):  14%|█▍        | 8/58 [00:03<00:18,  2.63it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=41.58 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.88it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.64 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.88it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=41.64 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.07it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.18 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.07it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.18 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.28it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.69 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.28it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=41.69 GB):  21%|██        | 12/58 [00:04<00:13,  3.48it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.74 GB):  21%|██        | 12/58 [00:04<00:13,  3.48it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=41.74 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.76it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.18 GB):  22%|██▏       | 13/58 [00:04<00:11,  3.76it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=42.18 GB):  24%|██▍       | 14/58 [00:05<00:10,  4.05it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.18 GB):  24%|██▍       | 14/58 [00:05<00:10,  4.05it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.18 GB):  26%|██▌       | 15/58 [00:05<00:09,  4.41it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.79 GB):  26%|██▌       | 15/58 [00:05<00:09,  4.41it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=41.79 GB):  28%|██▊       | 16/58 [00:05<00:08,  4.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.81 GB):  28%|██▊       | 16/58 [00:05<00:08,  4.93it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.81 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.37it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.16 GB):  29%|██▉       | 17/58 [00:05<00:07,  5.37it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=42.16 GB):  31%|███       | 18/58 [00:05<00:07,  5.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.16 GB):  31%|███       | 18/58 [00:05<00:07,  5.37it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.16 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.84it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.16 GB):  33%|███▎      | 19/58 [00:05<00:06,  5.84it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=42.16 GB):  34%|███▍      | 20/58 [00:06<00:06,  6.12it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=41.87 GB):  34%|███▍      | 20/58 [00:07<00:06,  6.12it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.87 GB):  36%|███▌      | 21/58 [00:07<00:20,  1.80it/s]Capturing num tokens (num_tokens=960 avail_mem=41.89 GB):  36%|███▌      | 21/58 [00:07<00:20,  1.80it/s] Capturing num tokens (num_tokens=896 avail_mem=41.91 GB):  36%|███▌      | 21/58 [00:07<00:20,  1.80it/s]

    Capturing num tokens (num_tokens=896 avail_mem=41.91 GB):  40%|███▉      | 23/58 [00:07<00:11,  2.93it/s]Capturing num tokens (num_tokens=832 avail_mem=42.12 GB):  40%|███▉      | 23/58 [00:07<00:11,  2.93it/s]Capturing num tokens (num_tokens=768 avail_mem=42.12 GB):  40%|███▉      | 23/58 [00:07<00:11,  2.93it/s]Capturing num tokens (num_tokens=768 avail_mem=42.12 GB):  43%|████▎     | 25/58 [00:07<00:07,  4.21it/s]Capturing num tokens (num_tokens=704 avail_mem=42.11 GB):  43%|████▎     | 25/58 [00:07<00:07,  4.21it/s]

    Capturing num tokens (num_tokens=640 avail_mem=42.11 GB):  43%|████▎     | 25/58 [00:07<00:07,  4.21it/s]Capturing num tokens (num_tokens=640 avail_mem=42.11 GB):  47%|████▋     | 27/58 [00:07<00:05,  5.75it/s]Capturing num tokens (num_tokens=576 avail_mem=41.98 GB):  47%|████▋     | 27/58 [00:07<00:05,  5.75it/s]Capturing num tokens (num_tokens=512 avail_mem=41.97 GB):  47%|████▋     | 27/58 [00:08<00:05,  5.75it/s]Capturing num tokens (num_tokens=512 avail_mem=41.97 GB):  50%|█████     | 29/58 [00:08<00:03,  7.47it/s]Capturing num tokens (num_tokens=480 avail_mem=42.08 GB):  50%|█████     | 29/58 [00:08<00:03,  7.47it/s]

    Capturing num tokens (num_tokens=448 avail_mem=42.07 GB):  50%|█████     | 29/58 [00:08<00:03,  7.47it/s]Capturing num tokens (num_tokens=448 avail_mem=42.07 GB):  53%|█████▎    | 31/58 [00:08<00:02,  9.12it/s]Capturing num tokens (num_tokens=416 avail_mem=42.06 GB):  53%|█████▎    | 31/58 [00:08<00:02,  9.12it/s]Capturing num tokens (num_tokens=384 avail_mem=42.06 GB):  53%|█████▎    | 31/58 [00:08<00:02,  9.12it/s]Capturing num tokens (num_tokens=384 avail_mem=42.06 GB):  57%|█████▋    | 33/58 [00:08<00:02, 10.95it/s]Capturing num tokens (num_tokens=352 avail_mem=42.05 GB):  57%|█████▋    | 33/58 [00:08<00:02, 10.95it/s]

    Capturing num tokens (num_tokens=320 avail_mem=42.04 GB):  57%|█████▋    | 33/58 [00:08<00:02, 10.95it/s]Capturing num tokens (num_tokens=288 avail_mem=42.04 GB):  57%|█████▋    | 33/58 [00:08<00:02, 10.95it/s]Capturing num tokens (num_tokens=288 avail_mem=42.04 GB):  62%|██████▏   | 36/58 [00:08<00:01, 13.84it/s]Capturing num tokens (num_tokens=256 avail_mem=41.99 GB):  62%|██████▏   | 36/58 [00:08<00:01, 13.84it/s]Capturing num tokens (num_tokens=240 avail_mem=42.02 GB):  62%|██████▏   | 36/58 [00:08<00:01, 13.84it/s]Capturing num tokens (num_tokens=224 avail_mem=42.02 GB):  62%|██████▏   | 36/58 [00:08<00:01, 13.84it/s]

    Capturing num tokens (num_tokens=224 avail_mem=42.02 GB):  67%|██████▋   | 39/58 [00:08<00:01, 15.48it/s]Capturing num tokens (num_tokens=208 avail_mem=42.01 GB):  67%|██████▋   | 39/58 [00:08<00:01, 15.48it/s]Capturing num tokens (num_tokens=192 avail_mem=42.00 GB):  67%|██████▋   | 39/58 [00:08<00:01, 15.48it/s]Capturing num tokens (num_tokens=192 avail_mem=42.00 GB):  71%|███████   | 41/58 [00:08<00:01, 16.39it/s]Capturing num tokens (num_tokens=176 avail_mem=42.00 GB):  71%|███████   | 41/58 [00:08<00:01, 16.39it/s]Capturing num tokens (num_tokens=160 avail_mem=41.99 GB):  71%|███████   | 41/58 [00:08<00:01, 16.39it/s]Capturing num tokens (num_tokens=144 avail_mem=41.98 GB):  71%|███████   | 41/58 [00:08<00:01, 16.39it/s]

    Capturing num tokens (num_tokens=144 avail_mem=41.98 GB):  76%|███████▌  | 44/58 [00:08<00:00, 18.58it/s]Capturing num tokens (num_tokens=128 avail_mem=41.98 GB):  76%|███████▌  | 44/58 [00:08<00:00, 18.58it/s]Capturing num tokens (num_tokens=112 avail_mem=41.94 GB):  76%|███████▌  | 44/58 [00:08<00:00, 18.58it/s]Capturing num tokens (num_tokens=96 avail_mem=41.97 GB):  76%|███████▌  | 44/58 [00:08<00:00, 18.58it/s] Capturing num tokens (num_tokens=96 avail_mem=41.97 GB):  81%|████████  | 47/58 [00:08<00:00, 20.94it/s]Capturing num tokens (num_tokens=80 avail_mem=41.95 GB):  81%|████████  | 47/58 [00:08<00:00, 20.94it/s]Capturing num tokens (num_tokens=64 avail_mem=41.95 GB):  81%|████████  | 47/58 [00:08<00:00, 20.94it/s]Capturing num tokens (num_tokens=48 avail_mem=41.94 GB):  81%|████████  | 47/58 [00:09<00:00, 20.94it/s]

    Capturing num tokens (num_tokens=48 avail_mem=41.94 GB):  86%|████████▌ | 50/58 [00:09<00:00, 22.67it/s]Capturing num tokens (num_tokens=32 avail_mem=41.93 GB):  86%|████████▌ | 50/58 [00:09<00:00, 22.67it/s]Capturing num tokens (num_tokens=28 avail_mem=41.93 GB):  86%|████████▌ | 50/58 [00:09<00:00, 22.67it/s]Capturing num tokens (num_tokens=24 avail_mem=41.92 GB):  86%|████████▌ | 50/58 [00:09<00:00, 22.67it/s]Capturing num tokens (num_tokens=24 avail_mem=41.92 GB):  91%|█████████▏| 53/58 [00:09<00:00, 24.00it/s]Capturing num tokens (num_tokens=20 avail_mem=41.92 GB):  91%|█████████▏| 53/58 [00:09<00:00, 24.00it/s]Capturing num tokens (num_tokens=16 avail_mem=41.91 GB):  91%|█████████▏| 53/58 [00:09<00:00, 24.00it/s]Capturing num tokens (num_tokens=12 avail_mem=41.91 GB):  91%|█████████▏| 53/58 [00:09<00:00, 24.00it/s]

    Capturing num tokens (num_tokens=12 avail_mem=41.91 GB):  97%|█████████▋| 56/58 [00:09<00:00, 25.07it/s]Capturing num tokens (num_tokens=8 avail_mem=41.90 GB):  97%|█████████▋| 56/58 [00:09<00:00, 25.07it/s] Capturing num tokens (num_tokens=4 avail_mem=41.89 GB):  97%|█████████▋| 56/58 [00:09<00:00, 25.07it/s]Capturing num tokens (num_tokens=4 avail_mem=41.89 GB): 100%|██████████| 58/58 [00:09<00:00,  6.19it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


Note that `--reasoning-parser` defines the parser used to interpret responses.

### OpenAI Compatible API

Using the OpenAI compatible API, the contract follows the [DeepSeek API design](https://api-docs.deepseek.com/guides/reasoning_model) established with the release of DeepSeek-R1:

- `reasoning_content`: The content of the CoT.
- `content`: The content of the final answer.


```python
# Initialize OpenAI-like client
client = OpenAI(api_key="None", base_url=f"http://0.0.0.0:{port}/v1")
model_name = client.models.list().data[0].id

messages = [
    {
        "role": "user",
        "content": "What is 1+3?",
    }
]
```

#### Non-Streaming Request


```python
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=False,  # Non-streaming
    extra_body={"separate_reasoning": True},
)
print_highlight("==== Reasoning ====")
print_highlight(response_non_stream.choices[0].message.reasoning_content)

print_highlight("==== Text ====")
print_highlight(response_non_stream.choices[0].message.content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   - First number: \(1\)<br>   - Second number: \(3\)<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**  <br>\[<br>\boxed{4}<br>\]</strong>


#### Streaming Request


```python
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=True,  # Non-streaming
    extra_body={"separate_reasoning": True},
)

reasoning_content = ""
content = ""
for chunk in response_stream:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content

print_highlight("==== Reasoning ====")
print_highlight(reasoning_content)

print_highlight("==== Text ====")
print_highlight(content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I arrive at the result, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>To find the sum of 1 and 3, follow these simple steps:<br><br>1. **Identify the numbers to add:**<br>   <br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>3. **Present the final answer:**<br>   <br>   \[<br>   \boxed{4}<br>   \]</strong>


Optionally, you can buffer the reasoning content to the last reasoning chunk (or the first chunk after the reasoning content).


```python
response_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=True,  # Non-streaming
    extra_body={"separate_reasoning": True, "stream_reasoning": False},
)

reasoning_content = ""
content = ""
for chunk in response_stream:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content

print_highlight("==== Reasoning ====")
print_highlight(reasoning_content)

print_highlight("==== Text ====")
print_highlight(content)
```


<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I perform the addition: 1 plus 3 equals 4.<br><br>Finally, I conclude that the result of 1 plus 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Problem:**<br>\[ 1 + 3 \]<br><br>**Solution:**<br><br>1. **Start with the first number:**<br>   \[<br>   1<br>   \]<br><br>2. **Add the second number:**<br>   \[<br>   1 + 3<br>   \]<br><br>3. **Calculate the sum:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


The reasoning separation is enable by default when specify . 
**To disable it, set the `separate_reasoning` option to `False` in request.**


```python
response_non_stream = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.6,
    top_p=0.95,
    stream=False,  # Non-streaming
    extra_body={"separate_reasoning": False},
)

print_highlight("==== Original Output ====")
print_highlight(response_non_stream.choices[0].message.content)
```


<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


### SGLang Native API 


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)

gen_url = f"http://localhost:{port}/generate"
gen_data = {
    "text": input,
    "sampling_params": {
        "skip_special_tokens": False,
        "max_new_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
    },
}
gen_response = requests.post(gen_url, json=gen_data).json()["text"]

print_highlight("==== Original Output ====")
print_highlight(gen_response)

parse_url = f"http://localhost:{port}/separate_reasoning"
separate_reasoning_data = {
    "text": gen_response,
    "reasoning_parser": "deepseek-r1",
}
separate_reasoning_response_json = requests.post(
    parse_url, json=separate_reasoning_data
).json()
print_highlight("==== Reasoning ====")
print_highlight(separate_reasoning_response_json["reasoning_text"])
print_highlight("==== Text ====")
print_highlight(separate_reasoning_response_json["text"])
```


<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I need to identify the two numbers in the problem, which are 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Finally, the sum of 1 and 3 is 4.<br></think><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I need to identify the two numbers in the problem, which are 1 and 3.<br><br>Next, I'll add these two numbers together.<br><br>Finally, the sum of 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>**Solution:**<br><br>We are asked to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>



```python
terminate_process(server_process)
```

### Offline Engine API


```python
import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.utils import print_highlight

llm = sgl.Engine(model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
sampling_params = {
    "max_new_tokens": 1024,
    "skip_special_tokens": False,
    "temperature": 0.6,
    "top_p": 0.95,
}
result = llm.generate(prompt=input, sampling_params=sampling_params)

generated_text = result["text"]  # Assume there is only one prompt

print_highlight("==== Original Output ====")
print_highlight(generated_text)

parser = ReasoningParser("deepseek-r1")
reasoning_text, text = parser.parse_non_stream(generated_text)
print_highlight("==== Reasoning ====")
print_highlight(reasoning_text)
print_highlight("==== Text ====")
print_highlight(text)
```

    Multi-thread loading shards:   0% Completed | 0/2 [00:00<?, ?it/s]

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.50s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.42s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.43s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:45,  5.01s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:45,  5.01s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:02,  2.19s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:02,  2.19s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:10,  1.28s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:10,  1.28s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:45,  1.18it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:05<00:31,  1.68it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:05<00:31,  1.68it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:05<00:23,  2.24it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:05<00:23,  2.24it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:17,  2.91it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:17,  2.91it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:13,  3.65it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:13,  3.65it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:10,  4.52it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:10,  4.52it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:10,  4.52it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:07,  6.22it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:07,  6.22it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:07,  6.22it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:06<00:05,  7.78it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:06<00:05,  7.78it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:06<00:05,  7.78it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:06<00:04,  9.35it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:06<00:04,  9.35it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:06<00:04,  9.35it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:06<00:03, 11.29it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:06<00:03, 11.29it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:06<00:03, 11.29it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:06<00:03, 11.29it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:06<00:02, 14.59it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:06<00:02, 14.59it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.59it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.59it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.59it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 20.07it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 20.07it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 20.07it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 20.07it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 20.07it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 20.07it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 20.07it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 29.02it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 29.02it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 29.02it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 29.02it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 29.02it/s]Compiling num tokens (num_tokens=352):  59%|█████▊    | 34/58 [00:07<00:00, 31.73it/s]Compiling num tokens (num_tokens=320):  59%|█████▊    | 34/58 [00:07<00:00, 31.73it/s]

    Compiling num tokens (num_tokens=288):  59%|█████▊    | 34/58 [00:07<00:00, 31.73it/s]Compiling num tokens (num_tokens=256):  59%|█████▊    | 34/58 [00:07<00:00, 31.73it/s]Compiling num tokens (num_tokens=240):  59%|█████▊    | 34/58 [00:07<00:00, 31.73it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:07<00:00, 29.24it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:07<00:00, 29.24it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:07<00:00, 29.24it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:07<00:00, 29.24it/s]

    Compiling num tokens (num_tokens=176):  66%|██████▌   | 38/58 [00:07<00:00, 29.24it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:07<00:00, 27.90it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:07<00:00, 27.90it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:07<00:00, 27.90it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:07<00:00, 27.90it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:07<00:00, 27.90it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:07<00:00, 27.90it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:07<00:00, 27.90it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:07<00:00, 27.90it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:07<00:00, 37.15it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:07<00:00, 37.15it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:07<00:00, 37.15it/s]Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:07<00:00, 37.15it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:07<00:00, 37.15it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:07<00:00, 37.15it/s]

    Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:07<00:00, 37.15it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:07<00:00, 37.15it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:07<00:00, 37.15it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:07<00:00, 37.15it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.41it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=42.26 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=42.26 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]Capturing num tokens (num_tokens=7680 avail_mem=42.23 GB):   2%|▏         | 1/58 [00:00<00:16,  3.40it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=42.23 GB):   3%|▎         | 2/58 [00:00<00:15,  3.57it/s]Capturing num tokens (num_tokens=7168 avail_mem=42.23 GB):   3%|▎         | 2/58 [00:00<00:15,  3.57it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=42.23 GB):   5%|▌         | 3/58 [00:00<00:14,  3.81it/s]Capturing num tokens (num_tokens=6656 avail_mem=42.23 GB):   5%|▌         | 3/58 [00:00<00:14,  3.81it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=42.23 GB):   7%|▋         | 4/58 [00:01<00:13,  4.10it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.23 GB):   7%|▋         | 4/58 [00:01<00:13,  4.10it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.23 GB):   9%|▊         | 5/58 [00:01<00:12,  4.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.23 GB):   9%|▊         | 5/58 [00:01<00:12,  4.25it/s]Capturing num tokens (num_tokens=5632 avail_mem=42.23 GB):  10%|█         | 6/58 [00:01<00:11,  4.65it/s]Capturing num tokens (num_tokens=5120 avail_mem=42.22 GB):  10%|█         | 6/58 [00:01<00:11,  4.65it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=42.22 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.23 GB):  12%|█▏        | 7/58 [00:01<00:10,  4.93it/s]Capturing num tokens (num_tokens=4608 avail_mem=42.23 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.14it/s]Capturing num tokens (num_tokens=4096 avail_mem=42.23 GB):  14%|█▍        | 8/58 [00:01<00:09,  5.14it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=42.23 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.71it/s]Capturing num tokens (num_tokens=3840 avail_mem=42.23 GB):  16%|█▌        | 9/58 [00:01<00:08,  5.71it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=42.23 GB):  17%|█▋        | 10/58 [00:02<00:10,  4.59it/s]Capturing num tokens (num_tokens=3584 avail_mem=42.22 GB):  17%|█▋        | 10/58 [00:02<00:10,  4.59it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=42.22 GB):  19%|█▉        | 11/58 [00:02<00:11,  4.14it/s]Capturing num tokens (num_tokens=3328 avail_mem=42.22 GB):  19%|█▉        | 11/58 [00:02<00:11,  4.14it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=42.22 GB):  21%|██        | 12/58 [00:02<00:11,  4.08it/s]Capturing num tokens (num_tokens=3072 avail_mem=42.22 GB):  21%|██        | 12/58 [00:02<00:11,  4.08it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=42.22 GB):  22%|██▏       | 13/58 [00:03<00:11,  3.97it/s]Capturing num tokens (num_tokens=2816 avail_mem=42.22 GB):  22%|██▏       | 13/58 [00:03<00:11,  3.97it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=42.22 GB):  24%|██▍       | 14/58 [00:03<00:10,  4.12it/s]Capturing num tokens (num_tokens=2560 avail_mem=42.22 GB):  24%|██▍       | 14/58 [00:03<00:10,  4.12it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=42.22 GB):  26%|██▌       | 15/58 [00:03<00:10,  4.29it/s]Capturing num tokens (num_tokens=2304 avail_mem=42.21 GB):  26%|██▌       | 15/58 [00:03<00:10,  4.29it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=42.21 GB):  28%|██▊       | 16/58 [00:03<00:09,  4.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.21 GB):  28%|██▊       | 16/58 [00:03<00:09,  4.44it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.21 GB):  29%|██▉       | 17/58 [00:03<00:08,  4.75it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.21 GB):  29%|██▉       | 17/58 [00:03<00:08,  4.75it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=42.21 GB):  31%|███       | 18/58 [00:04<00:07,  5.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.20 GB):  31%|███       | 18/58 [00:04<00:07,  5.03it/s]Capturing num tokens (num_tokens=1536 avail_mem=42.20 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.36it/s]Capturing num tokens (num_tokens=1280 avail_mem=42.20 GB):  33%|███▎      | 19/58 [00:04<00:07,  5.36it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=42.20 GB):  34%|███▍      | 20/58 [00:04<00:06,  5.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.19 GB):  34%|███▍      | 20/58 [00:04<00:06,  5.69it/s]Capturing num tokens (num_tokens=1024 avail_mem=42.19 GB):  36%|███▌      | 21/58 [00:04<00:05,  6.18it/s]Capturing num tokens (num_tokens=960 avail_mem=42.19 GB):  36%|███▌      | 21/58 [00:04<00:05,  6.18it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=42.19 GB):  38%|███▊      | 22/58 [00:04<00:05,  6.59it/s]Capturing num tokens (num_tokens=896 avail_mem=42.18 GB):  38%|███▊      | 22/58 [00:04<00:05,  6.59it/s]Capturing num tokens (num_tokens=896 avail_mem=42.18 GB):  40%|███▉      | 23/58 [00:04<00:05,  6.83it/s]Capturing num tokens (num_tokens=832 avail_mem=42.18 GB):  40%|███▉      | 23/58 [00:04<00:05,  6.83it/s]

    Capturing num tokens (num_tokens=832 avail_mem=42.18 GB):  41%|████▏     | 24/58 [00:04<00:04,  7.18it/s]Capturing num tokens (num_tokens=768 avail_mem=42.18 GB):  41%|████▏     | 24/58 [00:04<00:04,  7.18it/s]Capturing num tokens (num_tokens=768 avail_mem=42.18 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.54it/s]Capturing num tokens (num_tokens=704 avail_mem=42.17 GB):  43%|████▎     | 25/58 [00:04<00:04,  7.54it/s]

    Capturing num tokens (num_tokens=704 avail_mem=42.17 GB):  45%|████▍     | 26/58 [00:05<00:04,  7.92it/s]Capturing num tokens (num_tokens=640 avail_mem=42.17 GB):  45%|████▍     | 26/58 [00:05<00:04,  7.92it/s]Capturing num tokens (num_tokens=640 avail_mem=42.17 GB):  47%|████▋     | 27/58 [00:05<00:03,  8.09it/s]Capturing num tokens (num_tokens=576 avail_mem=42.17 GB):  47%|████▋     | 27/58 [00:05<00:03,  8.09it/s]

    Capturing num tokens (num_tokens=576 avail_mem=42.17 GB):  48%|████▊     | 28/58 [00:05<00:03,  8.53it/s]Capturing num tokens (num_tokens=512 avail_mem=42.16 GB):  48%|████▊     | 28/58 [00:05<00:03,  8.53it/s]Capturing num tokens (num_tokens=480 avail_mem=42.16 GB):  48%|████▊     | 28/58 [00:05<00:03,  8.53it/s]

    Capturing num tokens (num_tokens=480 avail_mem=42.16 GB):  52%|█████▏    | 30/58 [00:05<00:03,  8.73it/s]Capturing num tokens (num_tokens=448 avail_mem=42.14 GB):  52%|█████▏    | 30/58 [00:05<00:03,  8.73it/s]Capturing num tokens (num_tokens=448 avail_mem=42.14 GB):  53%|█████▎    | 31/58 [00:05<00:03,  8.78it/s]Capturing num tokens (num_tokens=416 avail_mem=42.13 GB):  53%|█████▎    | 31/58 [00:05<00:03,  8.78it/s]

    Capturing num tokens (num_tokens=384 avail_mem=42.11 GB):  53%|█████▎    | 31/58 [00:05<00:03,  8.78it/s]Capturing num tokens (num_tokens=384 avail_mem=42.11 GB):  57%|█████▋    | 33/58 [00:05<00:02, 10.23it/s]Capturing num tokens (num_tokens=352 avail_mem=41.63 GB):  57%|█████▋    | 33/58 [00:05<00:02, 10.23it/s]Capturing num tokens (num_tokens=320 avail_mem=41.47 GB):  57%|█████▋    | 33/58 [00:05<00:02, 10.23it/s]

    Capturing num tokens (num_tokens=320 avail_mem=41.47 GB):  60%|██████    | 35/58 [00:05<00:02, 10.55it/s]Capturing num tokens (num_tokens=288 avail_mem=41.47 GB):  60%|██████    | 35/58 [00:05<00:02, 10.55it/s]Capturing num tokens (num_tokens=256 avail_mem=41.47 GB):  60%|██████    | 35/58 [00:06<00:02, 10.55it/s]Capturing num tokens (num_tokens=240 avail_mem=41.46 GB):  60%|██████    | 35/58 [00:06<00:02, 10.55it/s]Capturing num tokens (num_tokens=240 avail_mem=41.46 GB):  66%|██████▌   | 38/58 [00:06<00:01, 13.66it/s]Capturing num tokens (num_tokens=224 avail_mem=41.46 GB):  66%|██████▌   | 38/58 [00:06<00:01, 13.66it/s]Capturing num tokens (num_tokens=208 avail_mem=41.45 GB):  66%|██████▌   | 38/58 [00:06<00:01, 13.66it/s]

    Capturing num tokens (num_tokens=192 avail_mem=41.45 GB):  66%|██████▌   | 38/58 [00:06<00:01, 13.66it/s]Capturing num tokens (num_tokens=192 avail_mem=41.45 GB):  71%|███████   | 41/58 [00:06<00:01, 16.61it/s]Capturing num tokens (num_tokens=176 avail_mem=41.44 GB):  71%|███████   | 41/58 [00:06<00:01, 16.61it/s]Capturing num tokens (num_tokens=160 avail_mem=41.44 GB):  71%|███████   | 41/58 [00:06<00:01, 16.61it/s]Capturing num tokens (num_tokens=144 avail_mem=41.44 GB):  71%|███████   | 41/58 [00:06<00:01, 16.61it/s]Capturing num tokens (num_tokens=144 avail_mem=41.44 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.74it/s]Capturing num tokens (num_tokens=128 avail_mem=41.44 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.74it/s]Capturing num tokens (num_tokens=112 avail_mem=41.44 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.74it/s]

    Capturing num tokens (num_tokens=96 avail_mem=41.43 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.74it/s] Capturing num tokens (num_tokens=80 avail_mem=41.42 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.74it/s]Capturing num tokens (num_tokens=64 avail_mem=41.42 GB):  76%|███████▌  | 44/58 [00:06<00:00, 18.74it/s]Capturing num tokens (num_tokens=64 avail_mem=41.42 GB):  84%|████████▍ | 49/58 [00:06<00:00, 24.79it/s]Capturing num tokens (num_tokens=48 avail_mem=41.42 GB):  84%|████████▍ | 49/58 [00:06<00:00, 24.79it/s]Capturing num tokens (num_tokens=32 avail_mem=41.41 GB):  84%|████████▍ | 49/58 [00:06<00:00, 24.79it/s]Capturing num tokens (num_tokens=28 avail_mem=41.41 GB):  84%|████████▍ | 49/58 [00:06<00:00, 24.79it/s]Capturing num tokens (num_tokens=24 avail_mem=41.41 GB):  84%|████████▍ | 49/58 [00:06<00:00, 24.79it/s]Capturing num tokens (num_tokens=20 avail_mem=41.41 GB):  84%|████████▍ | 49/58 [00:06<00:00, 24.79it/s]Capturing num tokens (num_tokens=20 avail_mem=41.41 GB):  93%|█████████▎| 54/58 [00:06<00:00, 29.32it/s]Capturing num tokens (num_tokens=16 avail_mem=41.40 GB):  93%|█████████▎| 54/58 [00:06<00:00, 29.32it/s]

    Capturing num tokens (num_tokens=12 avail_mem=41.40 GB):  93%|█████████▎| 54/58 [00:06<00:00, 29.32it/s]Capturing num tokens (num_tokens=8 avail_mem=41.40 GB):  93%|█████████▎| 54/58 [00:06<00:00, 29.32it/s] Capturing num tokens (num_tokens=4 avail_mem=41.39 GB):  93%|█████████▎| 54/58 [00:06<00:00, 29.32it/s]Capturing num tokens (num_tokens=4 avail_mem=41.39 GB): 100%|██████████| 58/58 [00:06<00:00, 31.96it/s]Capturing num tokens (num_tokens=4 avail_mem=41.39 GB): 100%|██████████| 58/58 [00:06<00:00,  8.68it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>I will start by identifying the two numbers involved: 1 and 3.<br><br>Next, I will add these two numbers together to find their total.<br><br>Finally, I will conclude that the result of adding 1 and 3 is 4.<br></think><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of the numbers 1 and 3.<br><br>I will start by identifying the two numbers involved: 1 and 3.<br><br>Next, I will add these two numbers together to find their total.<br><br>Finally, I will conclude that the result of adding 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
