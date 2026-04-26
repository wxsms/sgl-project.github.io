# Structured Outputs

You can specify a JSON schema, [regular expression](https://en.wikipedia.org/wiki/Regular_expression) or [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) to constrain the model output. The model output will be guaranteed to follow the given constraints. Only one constraint parameter (`json_schema`, `regex`, or `ebnf`) can be specified for a request.

SGLang supports three grammar backends:

- [XGrammar](https://github.com/mlc-ai/xgrammar)(default): Supports JSON schema, regular expression, and EBNF constraints.
- [Outlines](https://github.com/dottxt-ai/outlines): Supports JSON schema and regular expression constraints.
- [Llguidance](https://github.com/guidance-ai/llguidance): Supports JSON schema, regular expression, and EBNF constraints.

We suggest using XGrammar for its better performance and utility. XGrammar currently uses the [GGML BNF format](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md). For more details, see [XGrammar technical overview](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar).

To use Outlines, simply add `--grammar-backend outlines` when launching the server.
To use llguidance, add `--grammar-backend llguidance`  when launching the server.
If no backend is specified, XGrammar will be used as the default.

For better output quality, **It's advisable to explicitly include instructions in the prompt to guide the model to generate the desired format.** For example, you can specify, 'Please generate the output in the following JSON format: ...'.


## OpenAI Compatible API


```python
import openai
import os

from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

os.environ["TOKENIZERS_PARALLELISM"] = "false"


server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
```

    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(
    [2026-04-26 19:14:31] No platform detected. Using base SRTPlatform with defaults.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-26 19:14:33] `torch_dtype` is deprecated! Use `dtype` instead!


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    [2026-04-26 19:14:34] `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    [2026-04-26 19:14:36] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    [2026-04-26 19:14:39] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-26 19:14:41] `torch_dtype` is deprecated! Use `dtype` instead!


    [2026-04-26 19:14:43] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-04-26 19:14:43] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-04-26 19:14:45] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [2026-04-26 19:14:46] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.30it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.18it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.15it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.55it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.40it/s]


    2026-04-26 19:14:53,075 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 19:14:53] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:34,  5.87s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:34,  5.87s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:23,  2.56s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:23,  2.56s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:22,  1.50s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:22,  1.50s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:53,  1.02it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:53,  1.02it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:36,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:36,  1.43it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:26,  1.93it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:26,  1.93it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:20,  2.52it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:20,  2.52it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:15,  3.18it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:15,  3.18it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:12,  3.91it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:12,  3.91it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:10,  4.66it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:10,  4.66it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:08,  5.53it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:08,  5.53it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:08,  5.53it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.19it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.19it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.19it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  8.84it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  8.84it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  8.84it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.52it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.52it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:08<00:03, 10.52it/s]Compiling num tokens (num_tokens=1536):  33%|███▎      | 19/58 [00:08<00:03, 11.92it/s]Compiling num tokens (num_tokens=1280):  33%|███▎      | 19/58 [00:08<00:03, 11.92it/s]Compiling num tokens (num_tokens=1024):  33%|███▎      | 19/58 [00:08<00:03, 11.92it/s]Compiling num tokens (num_tokens=960):  33%|███▎      | 19/58 [00:08<00:03, 11.92it/s] Compiling num tokens (num_tokens=960):  38%|███▊      | 22/58 [00:08<00:02, 15.28it/s]Compiling num tokens (num_tokens=896):  38%|███▊      | 22/58 [00:08<00:02, 15.28it/s]

    Compiling num tokens (num_tokens=832):  38%|███▊      | 22/58 [00:08<00:02, 15.28it/s]Compiling num tokens (num_tokens=768):  38%|███▊      | 22/58 [00:08<00:02, 15.28it/s]Compiling num tokens (num_tokens=704):  38%|███▊      | 22/58 [00:08<00:02, 15.28it/s]Compiling num tokens (num_tokens=640):  38%|███▊      | 22/58 [00:08<00:02, 15.28it/s]Compiling num tokens (num_tokens=640):  47%|████▋     | 27/58 [00:08<00:01, 22.46it/s]Compiling num tokens (num_tokens=576):  47%|████▋     | 27/58 [00:08<00:01, 22.46it/s]Compiling num tokens (num_tokens=512):  47%|████▋     | 27/58 [00:08<00:01, 22.46it/s]Compiling num tokens (num_tokens=480):  47%|████▋     | 27/58 [00:08<00:01, 22.46it/s]Compiling num tokens (num_tokens=448):  47%|████▋     | 27/58 [00:08<00:01, 22.46it/s]Compiling num tokens (num_tokens=416):  47%|████▋     | 27/58 [00:08<00:01, 22.46it/s]Compiling num tokens (num_tokens=384):  47%|████▋     | 27/58 [00:08<00:01, 22.46it/s]

    Compiling num tokens (num_tokens=384):  57%|█████▋    | 33/58 [00:08<00:00, 30.59it/s]Compiling num tokens (num_tokens=352):  57%|█████▋    | 33/58 [00:08<00:00, 30.59it/s]Compiling num tokens (num_tokens=320):  57%|█████▋    | 33/58 [00:08<00:00, 30.59it/s]Compiling num tokens (num_tokens=288):  57%|█████▋    | 33/58 [00:08<00:00, 30.59it/s]Compiling num tokens (num_tokens=256):  57%|█████▋    | 33/58 [00:08<00:00, 30.59it/s]Compiling num tokens (num_tokens=240):  57%|█████▋    | 33/58 [00:08<00:00, 30.59it/s]Compiling num tokens (num_tokens=224):  57%|█████▋    | 33/58 [00:08<00:00, 30.59it/s]Compiling num tokens (num_tokens=208):  57%|█████▋    | 33/58 [00:08<00:00, 30.59it/s]Compiling num tokens (num_tokens=208):  69%|██████▉   | 40/58 [00:08<00:00, 39.99it/s]Compiling num tokens (num_tokens=192):  69%|██████▉   | 40/58 [00:08<00:00, 39.99it/s]Compiling num tokens (num_tokens=176):  69%|██████▉   | 40/58 [00:08<00:00, 39.99it/s]Compiling num tokens (num_tokens=160):  69%|██████▉   | 40/58 [00:08<00:00, 39.99it/s]Compiling num tokens (num_tokens=144):  69%|██████▉   | 40/58 [00:08<00:00, 39.99it/s]Compiling num tokens (num_tokens=128):  69%|██████▉   | 40/58 [00:08<00:00, 39.99it/s]Compiling num tokens (num_tokens=112):  69%|██████▉   | 40/58 [00:08<00:00, 39.99it/s]Compiling num tokens (num_tokens=96):  69%|██████▉   | 40/58 [00:08<00:00, 39.99it/s] 

    Compiling num tokens (num_tokens=96):  81%|████████  | 47/58 [00:08<00:00, 46.64it/s]Compiling num tokens (num_tokens=80):  81%|████████  | 47/58 [00:08<00:00, 46.64it/s]Compiling num tokens (num_tokens=64):  81%|████████  | 47/58 [00:08<00:00, 46.64it/s]Compiling num tokens (num_tokens=48):  81%|████████  | 47/58 [00:08<00:00, 46.64it/s]Compiling num tokens (num_tokens=32):  81%|████████  | 47/58 [00:08<00:00, 46.64it/s]Compiling num tokens (num_tokens=28):  81%|████████  | 47/58 [00:08<00:00, 46.64it/s]Compiling num tokens (num_tokens=28):  90%|████████▉ | 52/58 [00:08<00:00, 43.92it/s]Compiling num tokens (num_tokens=24):  90%|████████▉ | 52/58 [00:08<00:00, 43.92it/s]Compiling num tokens (num_tokens=20):  90%|████████▉ | 52/58 [00:08<00:00, 43.92it/s]Compiling num tokens (num_tokens=16):  90%|████████▉ | 52/58 [00:08<00:00, 43.92it/s]Compiling num tokens (num_tokens=12):  90%|████████▉ | 52/58 [00:08<00:00, 43.92it/s]Compiling num tokens (num_tokens=8):  90%|████████▉ | 52/58 [00:08<00:00, 43.92it/s] Compiling num tokens (num_tokens=4):  90%|████████▉ | 52/58 [00:08<00:00, 43.92it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.53it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=26.61 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=26.61 GB):   2%|▏         | 1/58 [00:00<00:19,  2.90it/s]Capturing num tokens (num_tokens=7680 avail_mem=26.57 GB):   2%|▏         | 1/58 [00:00<00:19,  2.90it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=26.57 GB):   3%|▎         | 2/58 [00:00<00:18,  3.07it/s]Capturing num tokens (num_tokens=7168 avail_mem=26.57 GB):   3%|▎         | 2/58 [00:00<00:18,  3.07it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=26.57 GB):   5%|▌         | 3/58 [00:00<00:16,  3.31it/s]Capturing num tokens (num_tokens=6656 avail_mem=26.57 GB):   5%|▌         | 3/58 [00:00<00:16,  3.31it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=26.57 GB):   7%|▋         | 4/58 [00:01<00:15,  3.41it/s]Capturing num tokens (num_tokens=6144 avail_mem=24.64 GB):   7%|▋         | 4/58 [00:01<00:15,  3.41it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=24.64 GB):   9%|▊         | 5/58 [00:01<00:15,  3.42it/s]Capturing num tokens (num_tokens=5632 avail_mem=23.20 GB):   9%|▊         | 5/58 [00:01<00:15,  3.42it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=23.20 GB):  10%|█         | 6/58 [00:01<00:13,  3.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=23.20 GB):  10%|█         | 6/58 [00:01<00:13,  3.82it/s]Capturing num tokens (num_tokens=5120 avail_mem=23.20 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.18it/s]Capturing num tokens (num_tokens=4608 avail_mem=23.20 GB):  12%|█▏        | 7/58 [00:01<00:12,  4.18it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=23.20 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=23.20 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.62it/s]Capturing num tokens (num_tokens=4096 avail_mem=23.20 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.01it/s]Capturing num tokens (num_tokens=3840 avail_mem=23.15 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.01it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=23.15 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.44it/s]Capturing num tokens (num_tokens=3584 avail_mem=23.13 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.44it/s]Capturing num tokens (num_tokens=3584 avail_mem=23.13 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.96it/s]Capturing num tokens (num_tokens=3328 avail_mem=23.12 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.96it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=23.12 GB):  21%|██        | 12/58 [00:02<00:07,  6.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=23.12 GB):  21%|██        | 12/58 [00:02<00:07,  6.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=23.12 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.21it/s]Capturing num tokens (num_tokens=2816 avail_mem=23.12 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.21it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=23.11 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.21it/s]Capturing num tokens (num_tokens=2560 avail_mem=23.11 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.45it/s]Capturing num tokens (num_tokens=2304 avail_mem=23.09 GB):  26%|██▌       | 15/58 [00:02<00:05,  8.45it/s]Capturing num tokens (num_tokens=2304 avail_mem=23.09 GB):  28%|██▊       | 16/58 [00:03<00:04,  8.69it/s]Capturing num tokens (num_tokens=2048 avail_mem=23.07 GB):  28%|██▊       | 16/58 [00:03<00:04,  8.69it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=23.07 GB):  29%|██▉       | 17/58 [00:03<00:05,  7.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=23.07 GB):  29%|██▉       | 17/58 [00:03<00:05,  7.51it/s]Capturing num tokens (num_tokens=1792 avail_mem=23.07 GB):  31%|███       | 18/58 [00:03<00:05,  6.91it/s]Capturing num tokens (num_tokens=1536 avail_mem=23.06 GB):  31%|███       | 18/58 [00:03<00:05,  6.91it/s]

    Capturing num tokens (num_tokens=1536 avail_mem=23.06 GB):  33%|███▎      | 19/58 [00:03<00:05,  6.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=23.06 GB):  33%|███▎      | 19/58 [00:03<00:05,  6.72it/s]Capturing num tokens (num_tokens=1280 avail_mem=23.06 GB):  34%|███▍      | 20/58 [00:03<00:05,  6.90it/s]Capturing num tokens (num_tokens=1024 avail_mem=23.06 GB):  34%|███▍      | 20/58 [00:03<00:05,  6.90it/s]

    Capturing num tokens (num_tokens=1024 avail_mem=23.06 GB):  36%|███▌      | 21/58 [00:03<00:05,  7.18it/s]Capturing num tokens (num_tokens=960 avail_mem=23.04 GB):  36%|███▌      | 21/58 [00:03<00:05,  7.18it/s] Capturing num tokens (num_tokens=896 avail_mem=23.04 GB):  36%|███▌      | 21/58 [00:03<00:05,  7.18it/s]Capturing num tokens (num_tokens=896 avail_mem=23.04 GB):  40%|███▉      | 23/58 [00:03<00:04,  8.74it/s]Capturing num tokens (num_tokens=832 avail_mem=23.04 GB):  40%|███▉      | 23/58 [00:03<00:04,  8.74it/s]

    Capturing num tokens (num_tokens=832 avail_mem=23.04 GB):  41%|████▏     | 24/58 [00:04<00:03,  8.76it/s]Capturing num tokens (num_tokens=768 avail_mem=41.57 GB):  41%|████▏     | 24/58 [00:04<00:03,  8.76it/s]Capturing num tokens (num_tokens=768 avail_mem=41.57 GB):  43%|████▎     | 25/58 [00:04<00:03,  8.30it/s]Capturing num tokens (num_tokens=704 avail_mem=41.56 GB):  43%|████▎     | 25/58 [00:04<00:03,  8.30it/s]Capturing num tokens (num_tokens=640 avail_mem=41.56 GB):  43%|████▎     | 25/58 [00:04<00:03,  8.30it/s]

    Capturing num tokens (num_tokens=576 avail_mem=41.56 GB):  43%|████▎     | 25/58 [00:04<00:03,  8.30it/s]Capturing num tokens (num_tokens=576 avail_mem=41.56 GB):  48%|████▊     | 28/58 [00:04<00:02, 13.35it/s]Capturing num tokens (num_tokens=512 avail_mem=41.55 GB):  48%|████▊     | 28/58 [00:04<00:02, 13.35it/s]Capturing num tokens (num_tokens=480 avail_mem=41.55 GB):  48%|████▊     | 28/58 [00:04<00:02, 13.35it/s]Capturing num tokens (num_tokens=448 avail_mem=41.54 GB):  48%|████▊     | 28/58 [00:04<00:02, 13.35it/s]Capturing num tokens (num_tokens=416 avail_mem=41.54 GB):  48%|████▊     | 28/58 [00:04<00:02, 13.35it/s]Capturing num tokens (num_tokens=416 avail_mem=41.54 GB):  55%|█████▌    | 32/58 [00:04<00:01, 18.78it/s]Capturing num tokens (num_tokens=384 avail_mem=41.54 GB):  55%|█████▌    | 32/58 [00:04<00:01, 18.78it/s]Capturing num tokens (num_tokens=352 avail_mem=41.53 GB):  55%|█████▌    | 32/58 [00:04<00:01, 18.78it/s]

    Capturing num tokens (num_tokens=320 avail_mem=41.53 GB):  55%|█████▌    | 32/58 [00:04<00:01, 18.78it/s]Capturing num tokens (num_tokens=288 avail_mem=41.52 GB):  55%|█████▌    | 32/58 [00:04<00:01, 18.78it/s]Capturing num tokens (num_tokens=288 avail_mem=41.52 GB):  62%|██████▏   | 36/58 [00:04<00:00, 23.03it/s]Capturing num tokens (num_tokens=256 avail_mem=41.52 GB):  62%|██████▏   | 36/58 [00:04<00:00, 23.03it/s]Capturing num tokens (num_tokens=240 avail_mem=41.52 GB):  62%|██████▏   | 36/58 [00:04<00:00, 23.03it/s]Capturing num tokens (num_tokens=224 avail_mem=41.51 GB):  62%|██████▏   | 36/58 [00:04<00:00, 23.03it/s]Capturing num tokens (num_tokens=208 avail_mem=41.51 GB):  62%|██████▏   | 36/58 [00:04<00:00, 23.03it/s]Capturing num tokens (num_tokens=208 avail_mem=41.51 GB):  69%|██████▉   | 40/58 [00:04<00:00, 26.74it/s]Capturing num tokens (num_tokens=192 avail_mem=41.50 GB):  69%|██████▉   | 40/58 [00:04<00:00, 26.74it/s]Capturing num tokens (num_tokens=176 avail_mem=41.50 GB):  69%|██████▉   | 40/58 [00:04<00:00, 26.74it/s]

    Capturing num tokens (num_tokens=160 avail_mem=41.49 GB):  69%|██████▉   | 40/58 [00:04<00:00, 26.74it/s]Capturing num tokens (num_tokens=144 avail_mem=41.49 GB):  69%|██████▉   | 40/58 [00:04<00:00, 26.74it/s]Capturing num tokens (num_tokens=144 avail_mem=41.49 GB):  76%|███████▌  | 44/58 [00:04<00:00, 29.09it/s]Capturing num tokens (num_tokens=128 avail_mem=41.49 GB):  76%|███████▌  | 44/58 [00:04<00:00, 29.09it/s]Capturing num tokens (num_tokens=112 avail_mem=41.49 GB):  76%|███████▌  | 44/58 [00:04<00:00, 29.09it/s]Capturing num tokens (num_tokens=96 avail_mem=41.49 GB):  76%|███████▌  | 44/58 [00:04<00:00, 29.09it/s] Capturing num tokens (num_tokens=80 avail_mem=41.48 GB):  76%|███████▌  | 44/58 [00:04<00:00, 29.09it/s]Capturing num tokens (num_tokens=80 avail_mem=41.48 GB):  83%|████████▎ | 48/58 [00:04<00:00, 31.34it/s]Capturing num tokens (num_tokens=64 avail_mem=41.48 GB):  83%|████████▎ | 48/58 [00:04<00:00, 31.34it/s]Capturing num tokens (num_tokens=48 avail_mem=41.48 GB):  83%|████████▎ | 48/58 [00:04<00:00, 31.34it/s]

    Capturing num tokens (num_tokens=32 avail_mem=41.47 GB):  83%|████████▎ | 48/58 [00:04<00:00, 31.34it/s]Capturing num tokens (num_tokens=28 avail_mem=41.47 GB):  83%|████████▎ | 48/58 [00:04<00:00, 31.34it/s]Capturing num tokens (num_tokens=28 avail_mem=41.47 GB):  90%|████████▉ | 52/58 [00:05<00:00, 33.23it/s]Capturing num tokens (num_tokens=24 avail_mem=41.46 GB):  90%|████████▉ | 52/58 [00:05<00:00, 33.23it/s]Capturing num tokens (num_tokens=20 avail_mem=41.46 GB):  90%|████████▉ | 52/58 [00:05<00:00, 33.23it/s]Capturing num tokens (num_tokens=16 avail_mem=41.46 GB):  90%|████████▉ | 52/58 [00:05<00:00, 33.23it/s]Capturing num tokens (num_tokens=12 avail_mem=41.45 GB):  90%|████████▉ | 52/58 [00:05<00:00, 33.23it/s]Capturing num tokens (num_tokens=12 avail_mem=41.45 GB):  97%|█████████▋| 56/58 [00:05<00:00, 34.57it/s]Capturing num tokens (num_tokens=8 avail_mem=41.45 GB):  97%|█████████▋| 56/58 [00:05<00:00, 34.57it/s] Capturing num tokens (num_tokens=4 avail_mem=41.44 GB):  97%|█████████▋| 56/58 [00:05<00:00, 34.57it/s]

    Capturing num tokens (num_tokens=4 avail_mem=41.44 GB): 100%|██████████| 58/58 [00:05<00:00, 11.22it/s]


    [2026-04-26 19:15:09] Tokenizer loaded as generic TokenizersBackend for meta-llama/Meta-Llama-3.1-8B-Instruct, retrying with use_fast=False


    [2026-04-26 19:15:12] Tokenizer for meta-llama/Meta-Llama-3.1-8B-Instruct loaded as generic TokenizersBackend. Set --trust-remote-code to load the model-specific tokenizer.


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


### JSON

you can directly define a JSON schema or use [Pydantic](https://docs.pydantic.dev/latest/) to define and validate the response.

**Using Pydantic**


```python
from pydantic import BaseModel, Field


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Please generate the information of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "foo",
            # convert the pydantic model to json schema
            "schema": CapitalInfo.model_json_schema(),
        },
    },
)

response_content = response.choices[0].message.content
# validate the JSON response by the pydantic model
capital_info = CapitalInfo.model_validate_json(response_content)
print_highlight(f"Validated response: {capital_info.model_dump_json()}")
```


<strong style='color: #00008B;'>Validated response: {"name":"Paris","population":2147000}</strong>


**JSON Schema Directly**



```python
import json

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Give me the information of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "foo", "schema": json.loads(json_schema)},
    },
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>{"name": "Paris", "population": 2147000}</strong>


### EBNF


```python
ebnf_grammar = """
root ::= city | description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful geography bot."},
        {
            "role": "user",
            "content": "Give me the information of the capital of France.",
        },
    ],
    temperature=0,
    max_tokens=32,
    extra_body={"ebnf": ebnf_grammar},
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>Paris is the capital of France</strong>


### Regular expression


```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0,
    max_tokens=128,
    extra_body={"regex": "(Paris|London)"},
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'>Paris</strong>


### Structural Tag


```python
tool_get_current_weather = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, e.g. 'San Francisco'",
                },
                "state": {
                    "type": "string",
                    "description": "the two-letter abbreviation for the state that the city is"
                    " in, e.g. 'CA' which would mean 'California'",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city", "state", "unit"],
        },
    },
}

tool_get_current_date = {
    "type": "function",
    "function": {
        "name": "get_current_date",
        "description": "Get the current date and time for a given timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                }
            },
            "required": ["timezone"],
        },
    },
}

schema_get_current_weather = tool_get_current_weather["function"]["parameters"]
schema_get_current_date = tool_get_current_date["function"]["parameters"]


def get_messages():
    return [
        {
            "role": "system",
            "content": f"""
# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{tool_get_current_weather["function"]}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{tool_get_current_date["function"]}
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant.""",
        },
        {
            "role": "user",
            "content": "You are in New York. Please get the current date and time, and the weather.",
        },
    ]


messages = get_messages()

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    response_format={
        "type": "structural_tag",
        "structures": [
            {
                "begin": "<function=get_current_weather>",
                "schema": schema_get_current_weather,
                "end": "</function>",
            },
            {
                "begin": "<function=get_current_date>",
                "schema": schema_get_current_date,
                "end": "</function>",
            },
        ],
        "triggers": ["<function="],
    },
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function></strong>



```python
# Support for XGrammar latest structural tag format
# <https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html>
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=messages,
    response_format={
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<function="],
            "tags": [
                {
                    "begin": "<function=get_current_weather>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema_get_current_weather,
                    },
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema_get_current_date,
                    },
                    "end": "</function>",
                },
            ],
            "at_least_one": False,
            "stop_after_first": False,
        },
    },
)

print_highlight(response.choices[0].message.content)
```


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function><br><br>Sources:<br>1. get_current_date function <br>2. get_current_weather function</strong>


## Native API and SGLang Runtime (SRT)

### JSON

**Using Pydantic**


```python
import requests
import json
from pydantic import BaseModel, Field

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


# Make API request
messages = [
    {
        "role": "user",
        "content": "Here is the information of the capital of France in the JSON format.\n",
    }
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json.dumps(CapitalInfo.model_json_schema()),
        },
    },
)
print_highlight(response.json())


response_data = json.loads(response.json()["text"])
# validate the response by the pydantic model
capital_info = CapitalInfo.model_validate(response_data)
print_highlight(f"Validated response: {capital_info.model_dump_json()}")
```


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': 'aeb67b44a4da406283c7aae743a1b679', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.13404646096751094, 'response_sent_to_client_ts': 1777230922.936118}}</strong>



<strong style='color: #00008B;'>Validated response: {"name":"Paris","population":2147000}</strong>


**JSON Schema Directly**


```python
json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

# JSON
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "json_schema": json_schema,
        },
    },
)

print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': '988adfd0a2764e99a93dc8e4f5f8d496', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 49, 'cached_tokens_details': {'device': 49, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1679791840724647, 'response_sent_to_client_ts': 1777230923.111952}}</strong>


### EBNF


```python
messages = [
    {
        "role": "user",
        "content": "Give me the information of the capital of France.",
    }
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "max_new_tokens": 128,
            "temperature": 0,
            "n": 3,
            "ebnf": (
                "root ::= city | description\n"
                'city ::= "London" | "Paris" | "Berlin" | "Rome"\n'
                'description ::= city " is " status\n'
                'status ::= "the capital of " country\n'
                'country ::= "England" | "France" | "Germany" | "Italy"'
            ),
        },
        "stream": False,
        "return_logprob": False,
    },
)

print_highlight(response.json())
```


<strong style='color: #00008B;'>[{'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '47fdf27ae71748778e425b42b9c73904', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.07576652197167277, 'response_sent_to_client_ts': 1777230923.210602}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '28d372c367374fbdba56e85b5b93ea8b', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.07571561215445399, 'response_sent_to_client_ts': 1777230923.2106097}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '5dec400e3b634b7d9fe4f24cfd274a57', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.07567788194864988, 'response_sent_to_client_ts': 1777230923.2106125}}]</strong>


### Regular expression


```python
messages = [
    {
        "role": "user",
        "content": "Paris is the capital of",
    }
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "regex": "(France|England)",
        },
    },
)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'France', 'output_ids': [50100, 128009], 'meta_info': {'id': 'ab525b14ef094d3681ec492f67edd018', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 2, 'cached_tokens': 31, 'cached_tokens_details': {'device': 31, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.030700973235070705, 'response_sent_to_client_ts': 1777230923.2487347}}</strong>


### Structural Tag


```python
from transformers import AutoTokenizer

# generate an answer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
payload = {
    "text": text,
    "sampling_params": {
        "structural_tag": json.dumps(
            {
                "type": "structural_tag",
                "structures": [
                    {
                        "begin": "<function=get_current_weather>",
                        "schema": schema_get_current_weather,
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=get_current_date>",
                        "schema": schema_get_current_date,
                        "end": "</function>",
                    },
                ],
                "triggers": ["<function="],
            }
        )
    },
}


# Send POST request to the API endpoint
response = requests.post(f"http://localhost:{port}/generate", json=payload)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'France.', 'output_ids': [50100, 13, 128009], 'meta_info': {'id': '5d8cb9e8b8c747fcb9103a77bfd8fdcf', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 3, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.0860287039540708, 'response_sent_to_client_ts': 1777230924.9752257}}</strong>



```python
# Support for XGrammar latest structural tag format
# <https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html>
payload = {
    "text": text,
    "sampling_params": {
        "structural_tag": json.dumps(
            {
                "type": "structural_tag",
                "format": {
                    "type": "triggered_tags",
                    "triggers": ["<function="],
                    "tags": [
                        {
                            "begin": "<function=get_current_weather>",
                            "content": {
                                "type": "json_schema",
                                "json_schema": schema_get_current_weather,
                            },
                            "end": "</function>",
                        },
                        {
                            "begin": "<function=get_current_date>",
                            "content": {
                                "type": "json_schema",
                                "json_schema": schema_get_current_date,
                            },
                            "end": "</function>",
                        },
                    ],
                    "at_least_one": False,
                    "stop_after_first": False,
                },
            }
        )
    },
}


# Send POST request to the API endpoint
response = requests.post(f"http://localhost:{port}/generate", json=payload)
print_highlight(response.json())
```


<strong style='color: #00008B;'>{'text': 'France.', 'output_ids': [50100, 13, 128009], 'meta_info': {'id': '92e7bc7c925845fb9e5c242cd07fe458', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'total_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 3, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.03906580014154315, 'response_sent_to_client_ts': 1777230925.0230029}}</strong>



```python
terminate_process(server_process)
```

## Offline Engine API


```python
import sglang as sgl

llm = sgl.Engine(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct", grammar_backend="xgrammar"
)
```

    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!


    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).
    Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.9.1+cu130).


    No platform detected. Using base SRTPlatform with defaults.
    No platform detected. Using base SRTPlatform with defaults.


    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.
    `BaseImageProcessorFast` is deprecated. The `Fast` suffix for image processors has been removed; use `BaseImageProcessor` instead.


    `torch_dtype` is deprecated! Use `dtype` instead!
    [2026-04-26 19:15:35] `torch_dtype` is deprecated! Use `dtype` instead!


    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
    [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.28it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.17it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.15it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.54it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:02<00:00,  1.39it/s]


    2026-04-26 19:15:46,742 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-04-26 19:15:46] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:33,  5.86s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:33,  5.86s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:06<02:24,  2.58s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:06<02:24,  2.58s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:06<01:22,  1.50s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:06<01:22,  1.50s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:53,  1.01it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:53,  1.01it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:37,  1.43it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:27,  1.92it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:27,  1.92it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:07<00:20,  2.51it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:07<00:20,  2.51it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:07<00:15,  3.18it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:07<00:15,  3.18it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:07<00:12,  3.94it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:07<00:12,  3.94it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:09,  4.81it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:09,  4.81it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:08,  5.71it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:08,  5.71it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:08,  5.71it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.40it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.40it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.40it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.03it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.03it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.03it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.91it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.91it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:08<00:03, 10.91it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:08<00:03, 10.91it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:08<00:02, 14.15it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:08<00:02, 14.15it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:08<00:02, 14.15it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:08<00:02, 14.15it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:08<00:02, 14.15it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:08<00:01, 19.76it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:08<00:01, 19.76it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:08<00:01, 19.76it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:08<00:01, 19.76it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:08<00:01, 19.76it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:08<00:01, 19.76it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:08<00:01, 27.03it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:08<00:01, 27.03it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:08<00:01, 27.03it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:08<00:01, 27.03it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:08<00:01, 27.03it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:08<00:01, 27.03it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:08<00:01, 27.03it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:08<00:00, 34.75it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:08<00:00, 34.75it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:08<00:00, 34.75it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:08<00:00, 34.75it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:08<00:00, 34.75it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:08<00:00, 34.75it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:08<00:00, 34.75it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:08<00:00, 34.75it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:08<00:00, 43.86it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:08<00:00, 43.86it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:08<00:00, 43.86it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:08<00:00, 43.86it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:08<00:00, 43.86it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:08<00:00, 43.86it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:08<00:00, 43.86it/s]Compiling num tokens (num_tokens=64):  72%|███████▏  | 42/58 [00:08<00:00, 43.86it/s]Compiling num tokens (num_tokens=64):  84%|████████▍ | 49/58 [00:08<00:00, 50.68it/s]Compiling num tokens (num_tokens=48):  84%|████████▍ | 49/58 [00:08<00:00, 50.68it/s]Compiling num tokens (num_tokens=32):  84%|████████▍ | 49/58 [00:08<00:00, 50.68it/s]

    Compiling num tokens (num_tokens=28):  84%|████████▍ | 49/58 [00:08<00:00, 50.68it/s]Compiling num tokens (num_tokens=24):  84%|████████▍ | 49/58 [00:08<00:00, 50.68it/s]Compiling num tokens (num_tokens=20):  84%|████████▍ | 49/58 [00:08<00:00, 50.68it/s]Compiling num tokens (num_tokens=16):  84%|████████▍ | 49/58 [00:08<00:00, 50.68it/s]Compiling num tokens (num_tokens=12):  84%|████████▍ | 49/58 [00:08<00:00, 50.68it/s]Compiling num tokens (num_tokens=8):  84%|████████▍ | 49/58 [00:08<00:00, 50.68it/s] Compiling num tokens (num_tokens=4):  84%|████████▍ | 49/58 [00:08<00:00, 50.68it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.65it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=10.71 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=10.71 GB):   2%|▏         | 1/58 [00:00<00:19,  2.88it/s]Capturing num tokens (num_tokens=7680 avail_mem=10.68 GB):   2%|▏         | 1/58 [00:00<00:19,  2.88it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=10.68 GB):   3%|▎         | 2/58 [00:00<00:18,  3.01it/s]Capturing num tokens (num_tokens=7168 avail_mem=10.67 GB):   3%|▎         | 2/58 [00:00<00:18,  3.01it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=10.67 GB):   5%|▌         | 3/58 [00:00<00:16,  3.24it/s]Capturing num tokens (num_tokens=6656 avail_mem=10.64 GB):   5%|▌         | 3/58 [00:00<00:16,  3.24it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=10.64 GB):   7%|▋         | 4/58 [00:01<00:15,  3.42it/s]Capturing num tokens (num_tokens=6144 avail_mem=10.64 GB):   7%|▋         | 4/58 [00:01<00:15,  3.42it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=10.64 GB):   9%|▊         | 5/58 [00:01<00:14,  3.67it/s]Capturing num tokens (num_tokens=5632 avail_mem=10.64 GB):   9%|▊         | 5/58 [00:01<00:14,  3.67it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=10.64 GB):  10%|█         | 6/58 [00:01<00:12,  4.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=10.64 GB):  10%|█         | 6/58 [00:01<00:12,  4.00it/s]Capturing num tokens (num_tokens=5120 avail_mem=10.64 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.31it/s]Capturing num tokens (num_tokens=4608 avail_mem=10.64 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.31it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=10.64 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.69it/s]Capturing num tokens (num_tokens=4096 avail_mem=10.61 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.69it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=10.61 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=10.56 GB):  16%|█▌        | 9/58 [00:02<00:10,  4.59it/s]Capturing num tokens (num_tokens=3840 avail_mem=10.56 GB):  17%|█▋        | 10/58 [00:02<00:09,  4.81it/s]Capturing num tokens (num_tokens=3584 avail_mem=10.55 GB):  17%|█▋        | 10/58 [00:02<00:09,  4.81it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=10.55 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=10.55 GB):  19%|█▉        | 11/58 [00:02<00:08,  5.25it/s]Capturing num tokens (num_tokens=3328 avail_mem=10.55 GB):  21%|██        | 12/58 [00:02<00:07,  5.93it/s]Capturing num tokens (num_tokens=3072 avail_mem=10.55 GB):  21%|██        | 12/58 [00:02<00:07,  5.93it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=10.55 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=10.55 GB):  22%|██▏       | 13/58 [00:02<00:06,  6.67it/s]Capturing num tokens (num_tokens=2816 avail_mem=10.55 GB):  24%|██▍       | 14/58 [00:02<00:06,  7.27it/s]Capturing num tokens (num_tokens=2560 avail_mem=10.54 GB):  24%|██▍       | 14/58 [00:02<00:06,  7.27it/s]

    Capturing num tokens (num_tokens=2304 avail_mem=10.54 GB):  24%|██▍       | 14/58 [00:03<00:06,  7.27it/s]Capturing num tokens (num_tokens=2304 avail_mem=10.54 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.41it/s]Capturing num tokens (num_tokens=2048 avail_mem=10.54 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.41it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=10.54 GB):  28%|██▊       | 16/58 [00:03<00:05,  7.41it/s]Capturing num tokens (num_tokens=1792 avail_mem=10.54 GB):  31%|███       | 18/58 [00:03<00:04,  9.30it/s]Capturing num tokens (num_tokens=1536 avail_mem=10.53 GB):  31%|███       | 18/58 [00:03<00:04,  9.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=10.53 GB):  31%|███       | 18/58 [00:03<00:04,  9.30it/s]Capturing num tokens (num_tokens=1280 avail_mem=10.53 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.51it/s]Capturing num tokens (num_tokens=1024 avail_mem=10.53 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.51it/s]Capturing num tokens (num_tokens=960 avail_mem=10.51 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.51it/s] 

    Capturing num tokens (num_tokens=896 avail_mem=10.51 GB):  34%|███▍      | 20/58 [00:03<00:03, 11.51it/s]Capturing num tokens (num_tokens=896 avail_mem=10.51 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.01it/s]Capturing num tokens (num_tokens=832 avail_mem=10.51 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.01it/s]Capturing num tokens (num_tokens=768 avail_mem=10.50 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.01it/s]Capturing num tokens (num_tokens=704 avail_mem=10.50 GB):  40%|███▉      | 23/58 [00:03<00:02, 15.01it/s]Capturing num tokens (num_tokens=704 avail_mem=10.50 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.95it/s]Capturing num tokens (num_tokens=640 avail_mem=10.49 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.95it/s]Capturing num tokens (num_tokens=576 avail_mem=10.49 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.95it/s]

    Capturing num tokens (num_tokens=512 avail_mem=10.49 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.95it/s]Capturing num tokens (num_tokens=480 avail_mem=10.48 GB):  45%|████▍     | 26/58 [00:03<00:01, 17.95it/s]Capturing num tokens (num_tokens=480 avail_mem=10.48 GB):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Capturing num tokens (num_tokens=448 avail_mem=10.48 GB):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Capturing num tokens (num_tokens=416 avail_mem=10.48 GB):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Capturing num tokens (num_tokens=384 avail_mem=10.48 GB):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Capturing num tokens (num_tokens=352 avail_mem=10.47 GB):  52%|█████▏    | 30/58 [00:03<00:01, 21.82it/s]Capturing num tokens (num_tokens=352 avail_mem=10.47 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.62it/s]Capturing num tokens (num_tokens=320 avail_mem=10.47 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.62it/s]

    Capturing num tokens (num_tokens=288 avail_mem=10.46 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.62it/s]Capturing num tokens (num_tokens=256 avail_mem=10.46 GB):  59%|█████▊    | 34/58 [00:03<00:00, 24.62it/s]Capturing num tokens (num_tokens=240 avail_mem=10.45 GB):  59%|█████▊    | 34/58 [00:04<00:00, 24.62it/s]Capturing num tokens (num_tokens=240 avail_mem=10.45 GB):  66%|██████▌   | 38/58 [00:04<00:00, 26.90it/s]Capturing num tokens (num_tokens=224 avail_mem=10.45 GB):  66%|██████▌   | 38/58 [00:04<00:00, 26.90it/s]Capturing num tokens (num_tokens=208 avail_mem=10.44 GB):  66%|██████▌   | 38/58 [00:04<00:00, 26.90it/s]Capturing num tokens (num_tokens=192 avail_mem=10.44 GB):  66%|██████▌   | 38/58 [00:04<00:00, 26.90it/s]Capturing num tokens (num_tokens=176 avail_mem=10.44 GB):  66%|██████▌   | 38/58 [00:04<00:00, 26.90it/s]

    Capturing num tokens (num_tokens=176 avail_mem=10.44 GB):  72%|███████▏  | 42/58 [00:04<00:00, 29.33it/s]Capturing num tokens (num_tokens=160 avail_mem=10.43 GB):  72%|███████▏  | 42/58 [00:04<00:00, 29.33it/s]Capturing num tokens (num_tokens=144 avail_mem=10.43 GB):  72%|███████▏  | 42/58 [00:04<00:00, 29.33it/s]Capturing num tokens (num_tokens=128 avail_mem=10.42 GB):  72%|███████▏  | 42/58 [00:04<00:00, 29.33it/s]Capturing num tokens (num_tokens=112 avail_mem=10.43 GB):  72%|███████▏  | 42/58 [00:04<00:00, 29.33it/s]Capturing num tokens (num_tokens=112 avail_mem=10.43 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.74it/s]Capturing num tokens (num_tokens=96 avail_mem=10.42 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.74it/s] Capturing num tokens (num_tokens=80 avail_mem=10.42 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.74it/s]Capturing num tokens (num_tokens=64 avail_mem=10.41 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.74it/s]Capturing num tokens (num_tokens=48 avail_mem=10.41 GB):  79%|███████▉  | 46/58 [00:04<00:00, 31.74it/s]

    Capturing num tokens (num_tokens=48 avail_mem=10.41 GB):  86%|████████▌ | 50/58 [00:04<00:00, 33.07it/s]Capturing num tokens (num_tokens=32 avail_mem=10.41 GB):  86%|████████▌ | 50/58 [00:04<00:00, 33.07it/s]Capturing num tokens (num_tokens=28 avail_mem=10.40 GB):  86%|████████▌ | 50/58 [00:04<00:00, 33.07it/s]Capturing num tokens (num_tokens=24 avail_mem=10.40 GB):  86%|████████▌ | 50/58 [00:04<00:00, 33.07it/s]Capturing num tokens (num_tokens=20 avail_mem=10.39 GB):  86%|████████▌ | 50/58 [00:04<00:00, 33.07it/s]Capturing num tokens (num_tokens=20 avail_mem=10.39 GB):  93%|█████████▎| 54/58 [00:04<00:00, 33.83it/s]Capturing num tokens (num_tokens=16 avail_mem=10.39 GB):  93%|█████████▎| 54/58 [00:04<00:00, 33.83it/s]Capturing num tokens (num_tokens=12 avail_mem=10.39 GB):  93%|█████████▎| 54/58 [00:04<00:00, 33.83it/s]Capturing num tokens (num_tokens=8 avail_mem=10.38 GB):  93%|█████████▎| 54/58 [00:04<00:00, 33.83it/s] Capturing num tokens (num_tokens=4 avail_mem=10.38 GB):  93%|█████████▎| 54/58 [00:04<00:00, 33.83it/s]

    Capturing num tokens (num_tokens=4 avail_mem=10.38 GB): 100%|██████████| 58/58 [00:04<00:00, 34.40it/s]Capturing num tokens (num_tokens=4 avail_mem=10.38 GB): 100%|██████████| 58/58 [00:04<00:00, 12.63it/s]


### JSON

**Using Pydantic**


```python
import json
from pydantic import BaseModel, Field

prompts = [
    "Give me the information of the capital of China in the JSON format.",
    "Give me the information of the capital of France in the JSON format.",
    "Give me the information of the capital of Ireland in the JSON format.",
]


# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


sampling_params = {
    "temperature": 0.1,
    "top_p": 0.95,
    "json_schema": json.dumps(CapitalInfo.model_json_schema()),
}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}")  # validate the output by the pydantic model
    capital_info = CapitalInfo.model_validate_json(output["text"])
    print_highlight(f"Validated output: {capital_info.model_dump_json()}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of China in the JSON format.</strong>



<strong style='color: #00008B;'>Validated output: {"name":"Beijing","population":21500000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of France in the JSON format.</strong>



<strong style='color: #00008B;'>Validated output: {"name":"Paris","population":2141000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Ireland in the JSON format.</strong>



<strong style='color: #00008B;'>Validated output: {"name":"Dublin","population":527617}</strong>


**JSON Schema Directly**


```python
prompts = [
    "Give me the information of the capital of China in the JSON format.",
    "Give me the information of the capital of France in the JSON format.",
    "Give me the information of the capital of Ireland in the JSON format.",
]

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

sampling_params = {"temperature": 0.1, "top_p": 0.95, "json_schema": json_schema}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of China in the JSON format.<br>Generated text: {"name": "Beijing", "population": 21500000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of France in the JSON format.<br>Generated text: {"name": "Paris", "population": 2141000}</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Ireland in the JSON format.<br>Generated text: {"name": "Dublin", "population": 527617}</strong>


### EBNF



```python
prompts = [
    "Give me the information of the capital of France.",
    "Give me the information of the capital of Germany.",
    "Give me the information of the capital of Italy.",
]

sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "ebnf": (
        "root ::= city | description\n"
        'city ::= "London" | "Paris" | "Berlin" | "Rome"\n'
        'description ::= city " is " status\n'
        'status ::= "the capital of " country\n'
        'country ::= "England" | "France" | "Germany" | "Italy"'
    ),
}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of France.<br>Generated text: Paris is the capital of France</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Germany.<br>Generated text: Berlin is the capital of Germany</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Italy.<br>Generated text: Paris is the capital of Italy</strong>


### Regular expression


```python
prompts = [
    "Please provide information about London as a major global city:",
    "Please provide information about Paris as a major global city:",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95, "regex": "(France|England)"}

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Please provide information about London as a major global city:<br>Generated text: England</strong>



<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: Please provide information about Paris as a major global city:<br>Generated text: France</strong>


### Structural Tag


```python
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_dict=False
)
prompts = [text]


sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "structural_tag": json.dumps(
        {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": "<function=get_current_weather>",
                    "schema": schema_get_current_weather,
                    "end": "</function>",
                },
                {
                    "begin": "<function=get_current_date>",
                    "schema": schema_get_current_date,
                    "end": "</function>",
                },
            ],
            "triggers": ["<function="],
        }
    ),
}


# Send POST request to the API endpoint
outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|><br><br>Cutting Knowledge Date: December 2023<br>Today Date: 26 Jul 2024<br><br><|eot_id|><|start_header_id|>user<|end_header_id|><br><br>Paris is the capital of<|eot_id|><|start_header_id|>assistant<|end_header_id|><br><br><br>Generated text: France.</strong>



```python
# Support for XGrammar latest structural tag format
# <https://xgrammar.mlc.ai/docs/tutorials/structural_tag.html>
sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "structural_tag": json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": ["<function="],
                "tags": [
                    {
                        "begin": "<function=get_current_weather>",
                        "content": {
                            "type": "json_schema",
                            "json_schema": schema_get_current_weather,
                        },
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=get_current_date>",
                        "content": {
                            "type": "json_schema",
                            "json_schema": schema_get_current_date,
                        },
                        "end": "</function>",
                    },
                ],
                "at_least_one": False,
                "stop_after_first": False,
            },
        }
    ),
}


# Send POST request to the API endpoint
outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print_highlight("===============================")
    print_highlight(f"Prompt: {prompt}\nGenerated text: {output['text']}")
```


<strong style='color: #00008B;'>===============================</strong>



<strong style='color: #00008B;'>Prompt: <|begin_of_text|><|start_header_id|>system<|end_header_id|><br><br>Cutting Knowledge Date: December 2023<br>Today Date: 26 Jul 2024<br><br><|eot_id|><|start_header_id|>user<|end_header_id|><br><br>Paris is the capital of<|eot_id|><|start_header_id|>assistant<|end_header_id|><br><br><br>Generated text: France.</strong>



```python
llm.shutdown()
```
