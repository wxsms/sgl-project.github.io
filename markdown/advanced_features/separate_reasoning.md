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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.36s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.26s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.28s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:55,  5.18s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:55,  5.18s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:06,  2.26s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:06,  2.26s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:12,  1.32s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:05<00:46,  1.15it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:05<00:46,  1.15it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:36,  1.46it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:27,  1.92it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:27,  1.92it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:20,  2.55it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:20,  2.55it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:15,  3.26it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:15,  3.26it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.11it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.11it/s]Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.11it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:06<00:08,  5.80it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:06<00:08,  5.80it/s]

    Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:06<00:08,  5.80it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.40it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.40it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.40it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.04it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.04it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.04it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.01it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.01it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.01it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.01it/s]

    Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.30it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.30it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.30it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.30it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.30it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]

    Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.18it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.18it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.18it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.18it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.18it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.18it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.18it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.18it/s]

    Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.18it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 46.23it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 46.23it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 46.23it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:07<00:00, 46.23it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:07<00:00, 46.23it/s]Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:07<00:00, 46.23it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:07<00:00, 46.23it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:07<00:00, 46.23it/s]Compiling num tokens (num_tokens=28):  76%|███████▌  | 44/58 [00:07<00:00, 46.23it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:07<00:00, 54.93it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:07<00:00, 54.93it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:07<00:00, 54.93it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:07<00:00, 54.93it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:07<00:00, 54.93it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:07<00:00, 54.93it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:07<00:00, 54.93it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:07<00:00,  7.26it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=44.18 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=44.18 GB):   2%|▏         | 1/58 [00:00<00:38,  1.49it/s]Capturing num tokens (num_tokens=7680 avail_mem=44.15 GB):   2%|▏         | 1/58 [00:00<00:38,  1.49it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=44.15 GB):   3%|▎         | 2/58 [00:01<00:51,  1.08it/s]Capturing num tokens (num_tokens=7168 avail_mem=44.15 GB):   3%|▎         | 2/58 [00:01<00:51,  1.08it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=44.15 GB):   5%|▌         | 3/58 [00:02<00:40,  1.36it/s]Capturing num tokens (num_tokens=6656 avail_mem=44.15 GB):   5%|▌         | 3/58 [00:02<00:40,  1.36it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=44.15 GB):   7%|▋         | 4/58 [00:02<00:33,  1.60it/s]Capturing num tokens (num_tokens=6144 avail_mem=44.15 GB):   7%|▋         | 4/58 [00:02<00:33,  1.60it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=44.15 GB):   9%|▊         | 5/58 [00:03<00:29,  1.81it/s]Capturing num tokens (num_tokens=5632 avail_mem=44.15 GB):   9%|▊         | 5/58 [00:03<00:29,  1.81it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=44.15 GB):  10%|█         | 6/58 [00:03<00:25,  2.07it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.15 GB):  10%|█         | 6/58 [00:03<00:25,  2.07it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=44.15 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.36it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.15 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.36it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=44.15 GB):  14%|█▍        | 8/58 [00:04<00:18,  2.72it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.15 GB):  14%|█▍        | 8/58 [00:04<00:18,  2.72it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=44.15 GB):  16%|█▌        | 9/58 [00:04<00:15,  3.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.15 GB):  16%|█▌        | 9/58 [00:04<00:15,  3.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.15 GB):  17%|█▋        | 10/58 [00:04<00:13,  3.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=44.14 GB):  17%|█▋        | 10/58 [00:04<00:13,  3.60it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=44.14 GB):  19%|█▉        | 11/58 [00:04<00:11,  4.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.14 GB):  19%|█▉        | 11/58 [00:04<00:11,  4.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.14 GB):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Capturing num tokens (num_tokens=3072 avail_mem=44.14 GB):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=44.14 GB):  21%|██        | 12/58 [00:04<00:09,  4.70it/s]Capturing num tokens (num_tokens=2816 avail_mem=44.14 GB):  24%|██▍       | 14/58 [00:04<00:06,  6.32it/s]Capturing num tokens (num_tokens=2560 avail_mem=44.13 GB):  24%|██▍       | 14/58 [00:04<00:06,  6.32it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.13 GB):  24%|██▍       | 14/58 [00:05<00:06,  6.32it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=44.13 GB):  28%|██▊       | 16/58 [00:05<00:05,  7.76it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.99 GB):  28%|██▊       | 16/58 [00:05<00:05,  7.76it/s]Capturing num tokens (num_tokens=2048 avail_mem=42.99 GB):  29%|██▉       | 17/58 [00:05<00:05,  7.07it/s]Capturing num tokens (num_tokens=1792 avail_mem=42.99 GB):  29%|██▉       | 17/58 [00:05<00:05,  7.07it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=42.99 GB):  31%|███       | 18/58 [00:05<00:05,  6.87it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.09 GB):  31%|███       | 18/58 [00:05<00:05,  6.87it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.09 GB):  33%|███▎      | 19/58 [00:05<00:05,  7.35it/s]Capturing num tokens (num_tokens=1280 avail_mem=43.10 GB):  33%|███▎      | 19/58 [00:05<00:05,  7.35it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=43.10 GB):  34%|███▍      | 20/58 [00:05<00:05,  7.16it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.09 GB):  34%|███▍      | 20/58 [00:05<00:05,  7.16it/s]Capturing num tokens (num_tokens=1024 avail_mem=43.09 GB):  36%|███▌      | 21/58 [00:05<00:05,  7.20it/s]Capturing num tokens (num_tokens=960 avail_mem=43.09 GB):  36%|███▌      | 21/58 [00:05<00:05,  7.20it/s] 

    Capturing num tokens (num_tokens=960 avail_mem=43.09 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.68it/s]Capturing num tokens (num_tokens=896 avail_mem=43.15 GB):  38%|███▊      | 22/58 [00:05<00:04,  7.68it/s]Capturing num tokens (num_tokens=896 avail_mem=43.15 GB):  40%|███▉      | 23/58 [00:06<00:04,  7.62it/s]Capturing num tokens (num_tokens=832 avail_mem=43.15 GB):  40%|███▉      | 23/58 [00:06<00:04,  7.62it/s]

    Capturing num tokens (num_tokens=832 avail_mem=43.15 GB):  41%|████▏     | 24/58 [00:06<00:04,  7.77it/s]Capturing num tokens (num_tokens=768 avail_mem=44.06 GB):  41%|████▏     | 24/58 [00:06<00:04,  7.77it/s]Capturing num tokens (num_tokens=768 avail_mem=44.06 GB):  43%|████▎     | 25/58 [00:06<00:03,  8.31it/s]Capturing num tokens (num_tokens=704 avail_mem=43.20 GB):  43%|████▎     | 25/58 [00:06<00:03,  8.31it/s]

    Capturing num tokens (num_tokens=704 avail_mem=43.20 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.23it/s]Capturing num tokens (num_tokens=640 avail_mem=43.20 GB):  45%|████▍     | 26/58 [00:06<00:03,  8.23it/s]Capturing num tokens (num_tokens=640 avail_mem=43.20 GB):  47%|████▋     | 27/58 [00:06<00:03,  8.40it/s]Capturing num tokens (num_tokens=576 avail_mem=44.05 GB):  47%|████▋     | 27/58 [00:06<00:03,  8.40it/s]

    Capturing num tokens (num_tokens=512 avail_mem=43.26 GB):  47%|████▋     | 27/58 [00:06<00:03,  8.40it/s]Capturing num tokens (num_tokens=512 avail_mem=43.26 GB):  50%|█████     | 29/58 [00:06<00:03,  8.65it/s]Capturing num tokens (num_tokens=480 avail_mem=43.26 GB):  50%|█████     | 29/58 [00:06<00:03,  8.65it/s]Capturing num tokens (num_tokens=448 avail_mem=44.04 GB):  50%|█████     | 29/58 [00:06<00:03,  8.65it/s]

    Capturing num tokens (num_tokens=448 avail_mem=44.04 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.15it/s]Capturing num tokens (num_tokens=416 avail_mem=43.31 GB):  53%|█████▎    | 31/58 [00:06<00:02,  9.15it/s]Capturing num tokens (num_tokens=416 avail_mem=43.31 GB):  55%|█████▌    | 32/58 [00:07<00:02,  9.28it/s]Capturing num tokens (num_tokens=384 avail_mem=44.03 GB):  55%|█████▌    | 32/58 [00:07<00:02,  9.28it/s]Capturing num tokens (num_tokens=352 avail_mem=43.37 GB):  55%|█████▌    | 32/58 [00:07<00:02,  9.28it/s]

    Capturing num tokens (num_tokens=352 avail_mem=43.37 GB):  59%|█████▊    | 34/58 [00:07<00:02,  9.66it/s]Capturing num tokens (num_tokens=320 avail_mem=43.36 GB):  59%|█████▊    | 34/58 [00:07<00:02,  9.66it/s]Capturing num tokens (num_tokens=288 avail_mem=44.03 GB):  59%|█████▊    | 34/58 [00:07<00:02,  9.66it/s]Capturing num tokens (num_tokens=288 avail_mem=44.03 GB):  62%|██████▏   | 36/58 [00:07<00:02, 10.40it/s]Capturing num tokens (num_tokens=256 avail_mem=43.43 GB):  62%|██████▏   | 36/58 [00:07<00:02, 10.40it/s]

    Capturing num tokens (num_tokens=240 avail_mem=44.02 GB):  62%|██████▏   | 36/58 [00:07<00:02, 10.40it/s]Capturing num tokens (num_tokens=240 avail_mem=44.02 GB):  66%|██████▌   | 38/58 [00:07<00:01, 10.73it/s]Capturing num tokens (num_tokens=224 avail_mem=43.49 GB):  66%|██████▌   | 38/58 [00:07<00:01, 10.73it/s]

    Capturing num tokens (num_tokens=208 avail_mem=44.01 GB):  66%|██████▌   | 38/58 [00:07<00:01, 10.73it/s]Capturing num tokens (num_tokens=208 avail_mem=44.01 GB):  69%|██████▉   | 40/58 [00:07<00:01, 10.44it/s]Capturing num tokens (num_tokens=192 avail_mem=43.51 GB):  69%|██████▉   | 40/58 [00:07<00:01, 10.44it/s]Capturing num tokens (num_tokens=176 avail_mem=44.00 GB):  69%|██████▉   | 40/58 [00:07<00:01, 10.44it/s]

    Capturing num tokens (num_tokens=176 avail_mem=44.00 GB):  72%|███████▏  | 42/58 [00:08<00:01, 10.62it/s]Capturing num tokens (num_tokens=160 avail_mem=43.54 GB):  72%|███████▏  | 42/58 [00:08<00:01, 10.62it/s]Capturing num tokens (num_tokens=144 avail_mem=43.68 GB):  72%|███████▏  | 42/58 [00:08<00:01, 10.62it/s]Capturing num tokens (num_tokens=144 avail_mem=43.68 GB):  76%|███████▌  | 44/58 [00:08<00:01, 11.16it/s]Capturing num tokens (num_tokens=128 avail_mem=44.00 GB):  76%|███████▌  | 44/58 [00:08<00:01, 11.16it/s]

    Capturing num tokens (num_tokens=112 avail_mem=43.59 GB):  76%|███████▌  | 44/58 [00:08<00:01, 11.16it/s]Capturing num tokens (num_tokens=112 avail_mem=43.59 GB):  79%|███████▉  | 46/58 [00:08<00:00, 12.00it/s]Capturing num tokens (num_tokens=96 avail_mem=43.99 GB):  79%|███████▉  | 46/58 [00:08<00:00, 12.00it/s] Capturing num tokens (num_tokens=80 avail_mem=43.61 GB):  79%|███████▉  | 46/58 [00:08<00:00, 12.00it/s]

    Capturing num tokens (num_tokens=80 avail_mem=43.61 GB):  83%|████████▎ | 48/58 [00:08<00:00, 12.88it/s]Capturing num tokens (num_tokens=64 avail_mem=43.98 GB):  83%|████████▎ | 48/58 [00:08<00:00, 12.88it/s]Capturing num tokens (num_tokens=48 avail_mem=43.64 GB):  83%|████████▎ | 48/58 [00:08<00:00, 12.88it/s]Capturing num tokens (num_tokens=48 avail_mem=43.64 GB):  86%|████████▌ | 50/58 [00:08<00:00, 13.76it/s]Capturing num tokens (num_tokens=32 avail_mem=43.66 GB):  86%|████████▌ | 50/58 [00:08<00:00, 13.76it/s]Capturing num tokens (num_tokens=28 avail_mem=43.97 GB):  86%|████████▌ | 50/58 [00:08<00:00, 13.76it/s]

    Capturing num tokens (num_tokens=28 avail_mem=43.97 GB):  90%|████████▉ | 52/58 [00:08<00:00, 14.10it/s]Capturing num tokens (num_tokens=24 avail_mem=43.69 GB):  90%|████████▉ | 52/58 [00:08<00:00, 14.10it/s]Capturing num tokens (num_tokens=20 avail_mem=43.94 GB):  90%|████████▉ | 52/58 [00:08<00:00, 14.10it/s]Capturing num tokens (num_tokens=20 avail_mem=43.94 GB):  93%|█████████▎| 54/58 [00:08<00:00, 15.01it/s]Capturing num tokens (num_tokens=16 avail_mem=43.95 GB):  93%|█████████▎| 54/58 [00:08<00:00, 15.01it/s]Capturing num tokens (num_tokens=12 avail_mem=43.74 GB):  93%|█████████▎| 54/58 [00:08<00:00, 15.01it/s]

    Capturing num tokens (num_tokens=8 avail_mem=43.76 GB):  93%|█████████▎| 54/58 [00:08<00:00, 15.01it/s] Capturing num tokens (num_tokens=8 avail_mem=43.76 GB):  98%|█████████▊| 57/58 [00:08<00:00, 16.72it/s]Capturing num tokens (num_tokens=4 avail_mem=43.92 GB):  98%|█████████▊| 57/58 [00:08<00:00, 16.72it/s]Capturing num tokens (num_tokens=4 avail_mem=43.92 GB): 100%|██████████| 58/58 [00:09<00:00,  6.44it/s]


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together.<br><br>Finally, I arrive at the result.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**  <br>   We have the numbers 1 and 3.<br><br>2. **Add the numbers together:**  <br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**  <br>\(\boxed{4}\)</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>I start by identifying the numbers involved in the addition: 1 and 3.<br><br>Next, I perform the addition by adding these two numbers together.<br><br>Finally, I calculate the result to find that 1 plus 3 equals 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>To solve the problem of adding 1 and 3, I start by identifying the two numbers involved.<br><br>Next, I perform the addition operation by summing these two numbers together.<br><br>Finally, I arrive at the result, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>To solve the addition problem \(1 + 3\), follow these simple steps:<br><br>1. **Start with the first number:**<br>   \[<br>   1<br>   \]<br><br>2. **Add the second number:**<br>   \[<br>   1 + 3<br>   \]<br><br>3. **Calculate the sum:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Final Answer:**<br>\[<br>\boxed{4}<br>\]</strong>


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



<strong style='color: #00008B;'>First, I recognize that the problem is asking for the sum of 1 and 3.<br><br>Next, I add the two numbers together: 1 plus 3 equals 4.<br><br>Therefore, the final answer is 4.<br></think><br><br>**Solution:**<br><br>We need to find the sum of 1 and 3.<br><br>\[<br>1 + 3 = 4<br>\]<br><br>**Answer:** \boxed{4}</strong>


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



<strong style='color: #00008B;'>First, I identify the two numbers that need to be added: 1 and 3.<br><br>Next, I add these numbers together to find the sum.<br><br>Finally, I arrive at the conclusion that the sum of 1 and 3 is 4.<br></think><br><br>Sure! Let's solve the addition step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**  <br>\[<br>\boxed{4}<br>\]</strong>


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:328: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      return await dependant.call(**values)



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I identify the two numbers that need to be added: 1 and 3.<br><br>Next, I add these numbers together to find the sum.<br><br>Finally, I arrive at the conclusion that the sum of 1 and 3 is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the addition step by step.<br><br>**Problem:**  <br>What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Add the numbers together:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**  <br>\[<br>\boxed{4}<br>\]</strong>



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

    Multi-thread loading shards:  50% Completed | 1/2 [00:01<00:01,  1.26s/it]

    Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.20s/it]Multi-thread loading shards: 100% Completed | 2/2 [00:02<00:00,  1.21s/it]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<04:48,  5.06s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:12,  2.36s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:12,  2.36s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:21,  1.48s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:21,  1.48s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:56,  1.05s/it]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:56,  1.05s/it]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:42,  1.25it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:42,  1.25it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:07<00:33,  1.56it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:07<00:33,  1.56it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:25,  1.99it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:25,  1.99it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:21,  2.36it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:21,  2.36it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:17,  2.76it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:17,  2.76it/s]

    Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:15,  3.17it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:15,  3.17it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:08<00:13,  3.56it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:08<00:13,  3.56it/s]Compiling num tokens (num_tokens=3328):  21%|██        | 12/58 [00:08<00:11,  4.04it/s]Compiling num tokens (num_tokens=3072):  21%|██        | 12/58 [00:08<00:11,  4.04it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:08<00:10,  4.46it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:08<00:10,  4.46it/s]Compiling num tokens (num_tokens=2816):  24%|██▍       | 14/58 [00:08<00:09,  4.88it/s]Compiling num tokens (num_tokens=2560):  24%|██▍       | 14/58 [00:08<00:09,  4.88it/s]

    Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:08<00:08,  5.35it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:08<00:08,  5.35it/s]Compiling num tokens (num_tokens=2304):  28%|██▊       | 16/58 [00:08<00:07,  5.92it/s]Compiling num tokens (num_tokens=2048):  28%|██▊       | 16/58 [00:08<00:07,  5.92it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:09<00:06,  6.52it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:09<00:06,  6.52it/s]Compiling num tokens (num_tokens=1792):  31%|███       | 18/58 [00:09<00:05,  7.20it/s]Compiling num tokens (num_tokens=1536):  31%|███       | 18/58 [00:09<00:05,  7.20it/s]

    Compiling num tokens (num_tokens=1280):  31%|███       | 18/58 [00:09<00:05,  7.20it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:09<00:04,  8.45it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:09<00:04,  8.45it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:09<00:04,  8.45it/s] 

    Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:09<00:03, 10.12it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:09<00:03, 10.12it/s]Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:09<00:03, 10.12it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:09<00:02, 11.66it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:09<00:02, 11.66it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:09<00:02, 11.66it/s]

    Compiling num tokens (num_tokens=704):  45%|████▍     | 26/58 [00:09<00:02, 13.35it/s]Compiling num tokens (num_tokens=640):  45%|████▍     | 26/58 [00:09<00:02, 13.35it/s]Compiling num tokens (num_tokens=576):  45%|████▍     | 26/58 [00:09<00:02, 13.35it/s]Compiling num tokens (num_tokens=512):  45%|████▍     | 26/58 [00:09<00:02, 13.35it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:09<00:01, 15.93it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:09<00:01, 15.93it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:09<00:01, 15.93it/s]

    Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:09<00:01, 15.93it/s]Compiling num tokens (num_tokens=416):  55%|█████▌    | 32/58 [00:09<00:01, 17.84it/s]Compiling num tokens (num_tokens=384):  55%|█████▌    | 32/58 [00:09<00:01, 17.84it/s]Compiling num tokens (num_tokens=352):  55%|█████▌    | 32/58 [00:10<00:01, 17.84it/s]Compiling num tokens (num_tokens=320):  55%|█████▌    | 32/58 [00:10<00:01, 17.84it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:10<00:01, 19.76it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:10<00:01, 19.76it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:10<00:01, 19.76it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:10<00:01, 19.76it/s]Compiling num tokens (num_tokens=240):  66%|██████▌   | 38/58 [00:10<00:00, 21.63it/s]Compiling num tokens (num_tokens=224):  66%|██████▌   | 38/58 [00:10<00:00, 21.63it/s]Compiling num tokens (num_tokens=208):  66%|██████▌   | 38/58 [00:10<00:00, 21.63it/s]Compiling num tokens (num_tokens=192):  66%|██████▌   | 38/58 [00:10<00:00, 21.63it/s]Compiling num tokens (num_tokens=192):  71%|███████   | 41/58 [00:10<00:00, 23.47it/s]Compiling num tokens (num_tokens=176):  71%|███████   | 41/58 [00:10<00:00, 23.47it/s]

    Compiling num tokens (num_tokens=160):  71%|███████   | 41/58 [00:10<00:00, 23.47it/s]Compiling num tokens (num_tokens=144):  71%|███████   | 41/58 [00:10<00:00, 23.47it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:10<00:00, 20.05it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:10<00:00, 20.05it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:10<00:00, 20.05it/s]

    Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:10<00:00, 20.05it/s] Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:10<00:00, 22.06it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:10<00:00, 22.06it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:10<00:00, 22.06it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:10<00:00, 22.06it/s]Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:10<00:00, 23.17it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:10<00:00, 23.17it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:10<00:00, 23.17it/s]

    Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:10<00:00, 23.17it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:10<00:00, 23.17it/s]Compiling num tokens (num_tokens=20):  93%|█████████▎| 54/58 [00:10<00:00, 26.33it/s]Compiling num tokens (num_tokens=16):  93%|█████████▎| 54/58 [00:10<00:00, 26.33it/s]Compiling num tokens (num_tokens=12):  93%|█████████▎| 54/58 [00:10<00:00, 26.33it/s]Compiling num tokens (num_tokens=8):  93%|█████████▎| 54/58 [00:10<00:00, 26.33it/s] Compiling num tokens (num_tokens=4):  93%|█████████▎| 54/58 [00:10<00:00, 26.33it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00, 29.52it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:10<00:00,  5.29it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.32 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.32 GB):   2%|▏         | 1/58 [00:00<00:36,  1.57it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.29 GB):   2%|▏         | 1/58 [00:00<00:36,  1.57it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.29 GB):   3%|▎         | 2/58 [00:01<00:32,  1.70it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.29 GB):   3%|▎         | 2/58 [00:01<00:32,  1.70it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.29 GB):   5%|▌         | 3/58 [00:01<00:30,  1.80it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.29 GB):   5%|▌         | 3/58 [00:01<00:30,  1.80it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.29 GB):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]Capturing num tokens (num_tokens=6144 avail_mem=43.29 GB):   7%|▋         | 4/58 [00:02<00:28,  1.92it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=43.29 GB):   9%|▊         | 5/58 [00:02<00:26,  2.03it/s]Capturing num tokens (num_tokens=5632 avail_mem=43.29 GB):   9%|▊         | 5/58 [00:02<00:26,  2.03it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=43.29 GB):  10%|█         | 6/58 [00:03<00:23,  2.18it/s]Capturing num tokens (num_tokens=5120 avail_mem=43.28 GB):  10%|█         | 6/58 [00:03<00:23,  2.18it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=43.28 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.33it/s]Capturing num tokens (num_tokens=4608 avail_mem=43.29 GB):  12%|█▏        | 7/58 [00:03<00:21,  2.33it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=43.29 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.52it/s]Capturing num tokens (num_tokens=4096 avail_mem=43.29 GB):  14%|█▍        | 8/58 [00:03<00:19,  2.52it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=43.29 GB):  16%|█▌        | 9/58 [00:04<00:18,  2.67it/s]Capturing num tokens (num_tokens=3840 avail_mem=43.29 GB):  16%|█▌        | 9/58 [00:04<00:18,  2.67it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=43.29 GB):  17%|█▋        | 10/58 [00:04<00:16,  2.87it/s]Capturing num tokens (num_tokens=3584 avail_mem=43.28 GB):  17%|█▋        | 10/58 [00:04<00:16,  2.87it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=43.28 GB):  19%|█▉        | 11/58 [00:04<00:15,  3.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.28 GB):  19%|█▉        | 11/58 [00:04<00:15,  3.07it/s]Capturing num tokens (num_tokens=3328 avail_mem=43.28 GB):  21%|██        | 12/58 [00:04<00:12,  3.80it/s]Capturing num tokens (num_tokens=3072 avail_mem=43.28 GB):  21%|██        | 12/58 [00:04<00:12,  3.80it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=43.28 GB):  22%|██▏       | 13/58 [00:04<00:09,  4.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.28 GB):  22%|██▏       | 13/58 [00:04<00:09,  4.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=43.28 GB):  24%|██▍       | 14/58 [00:04<00:08,  5.21it/s]Capturing num tokens (num_tokens=2560 avail_mem=61.63 GB):  24%|██▍       | 14/58 [00:04<00:08,  5.21it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=61.63 GB):  24%|██▍       | 14/58 [00:05<00:08,  5.21it/s]Capturing num tokens (num_tokens=2304 avail_mem=61.63 GB):  28%|██▊       | 16/58 [00:05<00:05,  7.19it/s]Capturing num tokens (num_tokens=2048 avail_mem=61.62 GB):  28%|██▊       | 16/58 [00:05<00:05,  7.19it/s]Capturing num tokens (num_tokens=1792 avail_mem=61.62 GB):  28%|██▊       | 16/58 [00:05<00:05,  7.19it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=61.62 GB):  31%|███       | 18/58 [00:05<00:04,  9.16it/s]Capturing num tokens (num_tokens=1536 avail_mem=61.62 GB):  31%|███       | 18/58 [00:05<00:04,  9.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.62 GB):  31%|███       | 18/58 [00:05<00:04,  9.16it/s]Capturing num tokens (num_tokens=1280 avail_mem=61.62 GB):  34%|███▍      | 20/58 [00:05<00:03, 11.36it/s]Capturing num tokens (num_tokens=1024 avail_mem=61.61 GB):  34%|███▍      | 20/58 [00:05<00:03, 11.36it/s]Capturing num tokens (num_tokens=960 avail_mem=61.60 GB):  34%|███▍      | 20/58 [00:05<00:03, 11.36it/s] Capturing num tokens (num_tokens=896 avail_mem=61.60 GB):  34%|███▍      | 20/58 [00:05<00:03, 11.36it/s]

    Capturing num tokens (num_tokens=896 avail_mem=61.60 GB):  40%|███▉      | 23/58 [00:05<00:02, 14.73it/s]Capturing num tokens (num_tokens=832 avail_mem=61.60 GB):  40%|███▉      | 23/58 [00:05<00:02, 14.73it/s]Capturing num tokens (num_tokens=768 avail_mem=61.59 GB):  40%|███▉      | 23/58 [00:05<00:02, 14.73it/s]Capturing num tokens (num_tokens=704 avail_mem=61.59 GB):  40%|███▉      | 23/58 [00:05<00:02, 14.73it/s]Capturing num tokens (num_tokens=704 avail_mem=61.59 GB):  45%|████▍     | 26/58 [00:05<00:01, 17.61it/s]Capturing num tokens (num_tokens=640 avail_mem=61.59 GB):  45%|████▍     | 26/58 [00:05<00:01, 17.61it/s]Capturing num tokens (num_tokens=576 avail_mem=61.58 GB):  45%|████▍     | 26/58 [00:05<00:01, 17.61it/s]Capturing num tokens (num_tokens=512 avail_mem=61.58 GB):  45%|████▍     | 26/58 [00:05<00:01, 17.61it/s]

    Capturing num tokens (num_tokens=512 avail_mem=61.58 GB):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Capturing num tokens (num_tokens=480 avail_mem=61.57 GB):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Capturing num tokens (num_tokens=448 avail_mem=61.57 GB):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Capturing num tokens (num_tokens=416 avail_mem=61.57 GB):  50%|█████     | 29/58 [00:05<00:01, 20.43it/s]Capturing num tokens (num_tokens=416 avail_mem=61.57 GB):  55%|█████▌    | 32/58 [00:05<00:01, 22.70it/s]Capturing num tokens (num_tokens=384 avail_mem=61.56 GB):  55%|█████▌    | 32/58 [00:05<00:01, 22.70it/s]Capturing num tokens (num_tokens=352 avail_mem=61.56 GB):  55%|█████▌    | 32/58 [00:05<00:01, 22.70it/s]Capturing num tokens (num_tokens=320 avail_mem=61.55 GB):  55%|█████▌    | 32/58 [00:05<00:01, 22.70it/s]Capturing num tokens (num_tokens=288 avail_mem=61.56 GB):  55%|█████▌    | 32/58 [00:05<00:01, 22.70it/s]

    Capturing num tokens (num_tokens=288 avail_mem=61.56 GB):  62%|██████▏   | 36/58 [00:05<00:00, 25.73it/s]Capturing num tokens (num_tokens=256 avail_mem=61.55 GB):  62%|██████▏   | 36/58 [00:05<00:00, 25.73it/s]Capturing num tokens (num_tokens=240 avail_mem=61.55 GB):  62%|██████▏   | 36/58 [00:05<00:00, 25.73it/s]Capturing num tokens (num_tokens=224 avail_mem=61.55 GB):  62%|██████▏   | 36/58 [00:05<00:00, 25.73it/s]Capturing num tokens (num_tokens=208 avail_mem=61.54 GB):  62%|██████▏   | 36/58 [00:06<00:00, 25.73it/s]Capturing num tokens (num_tokens=208 avail_mem=61.54 GB):  69%|██████▉   | 40/58 [00:06<00:00, 28.24it/s]Capturing num tokens (num_tokens=192 avail_mem=61.54 GB):  69%|██████▉   | 40/58 [00:06<00:00, 28.24it/s]Capturing num tokens (num_tokens=176 avail_mem=61.54 GB):  69%|██████▉   | 40/58 [00:06<00:00, 28.24it/s]Capturing num tokens (num_tokens=160 avail_mem=61.53 GB):  69%|██████▉   | 40/58 [00:06<00:00, 28.24it/s]

    Capturing num tokens (num_tokens=144 avail_mem=61.53 GB):  69%|██████▉   | 40/58 [00:06<00:00, 28.24it/s]Capturing num tokens (num_tokens=144 avail_mem=61.53 GB):  76%|███████▌  | 44/58 [00:06<00:00, 30.36it/s]Capturing num tokens (num_tokens=128 avail_mem=61.53 GB):  76%|███████▌  | 44/58 [00:06<00:00, 30.36it/s]Capturing num tokens (num_tokens=112 avail_mem=61.53 GB):  76%|███████▌  | 44/58 [00:06<00:00, 30.36it/s]Capturing num tokens (num_tokens=96 avail_mem=61.52 GB):  76%|███████▌  | 44/58 [00:06<00:00, 30.36it/s] Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  76%|███████▌  | 44/58 [00:06<00:00, 30.36it/s]Capturing num tokens (num_tokens=80 avail_mem=61.52 GB):  83%|████████▎ | 48/58 [00:06<00:00, 31.96it/s]Capturing num tokens (num_tokens=64 avail_mem=61.51 GB):  83%|████████▎ | 48/58 [00:06<00:00, 31.96it/s]Capturing num tokens (num_tokens=48 avail_mem=61.51 GB):  83%|████████▎ | 48/58 [00:06<00:00, 31.96it/s]Capturing num tokens (num_tokens=32 avail_mem=61.51 GB):  83%|████████▎ | 48/58 [00:06<00:00, 31.96it/s]

    Capturing num tokens (num_tokens=28 avail_mem=61.51 GB):  83%|████████▎ | 48/58 [00:06<00:00, 31.96it/s]Capturing num tokens (num_tokens=28 avail_mem=61.51 GB):  90%|████████▉ | 52/58 [00:06<00:00, 33.20it/s]Capturing num tokens (num_tokens=24 avail_mem=61.50 GB):  90%|████████▉ | 52/58 [00:06<00:00, 33.20it/s]Capturing num tokens (num_tokens=20 avail_mem=61.50 GB):  90%|████████▉ | 52/58 [00:06<00:00, 33.20it/s]Capturing num tokens (num_tokens=16 avail_mem=61.50 GB):  90%|████████▉ | 52/58 [00:06<00:00, 33.20it/s]Capturing num tokens (num_tokens=12 avail_mem=61.49 GB):  90%|████████▉ | 52/58 [00:06<00:00, 33.20it/s]Capturing num tokens (num_tokens=12 avail_mem=61.49 GB):  97%|█████████▋| 56/58 [00:06<00:00, 33.84it/s]Capturing num tokens (num_tokens=8 avail_mem=61.49 GB):  97%|█████████▋| 56/58 [00:06<00:00, 33.84it/s] Capturing num tokens (num_tokens=4 avail_mem=61.49 GB):  97%|█████████▋| 56/58 [00:06<00:00, 33.84it/s]Capturing num tokens (num_tokens=4 avail_mem=61.49 GB): 100%|██████████| 58/58 [00:06<00:00,  8.87it/s]



<strong style='color: #00008B;'>==== Original Output ====</strong>



<strong style='color: #00008B;'>First, I recognize that I need to add the numbers 1 and 3 together.<br><br>I start by identifying the two numbers involved in the addition, which are 1 and 3.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum, which is 4.<br></think><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



<strong style='color: #00008B;'>==== Reasoning ====</strong>



<strong style='color: #00008B;'>First, I recognize that I need to add the numbers 1 and 3 together.<br><br>I start by identifying the two numbers involved in the addition, which are 1 and 3.<br><br>Next, I perform the addition operation by combining these two numbers.<br><br>Finally, I calculate the sum, which is 4.<br></strong>



<strong style='color: #00008B;'>==== Text ====</strong>



<strong style='color: #00008B;'><br><br>Sure! Let's solve the addition problem step by step.<br><br>**Question:** What is \(1 + 3\)?<br><br>**Solution:**<br><br>1. **Identify the numbers to add:**<br>   \[<br>   1 \quad \text{and} \quad 3<br>   \]<br><br>2. **Perform the addition:**<br>   \[<br>   1 + 3 = 4<br>   \]<br><br>**Answer:**<br>\[<br>\boxed{4}<br>\]</strong>



```python
llm.shutdown()
```

## Supporting New Reasoning Model Schemas

For future reasoning models, you can implement the reasoning parser as a subclass of `BaseReasoningFormatDetector` in `python/sglang/srt/reasoning_parser.py` and specify the reasoning parser for new reasoning model schemas accordingly.
