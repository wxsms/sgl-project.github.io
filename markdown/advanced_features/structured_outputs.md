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

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.21it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.11it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.10it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.48it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.32it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:14,  5.52s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:14,  5.52s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:15,  2.42s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:15,  2.42s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:17,  1.41s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:17,  1.41s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:50,  1.07it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:50,  1.07it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:35,  1.50it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:25,  2.02it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:25,  2.02it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:19,  2.62it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:19,  2.62it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:15,  3.29it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:15,  3.29it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:12,  4.05it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:12,  4.05it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:07<00:09,  4.90it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:07<00:09,  4.90it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:08,  5.81it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:08,  5.81it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:08,  5.81it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:06,  7.46it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:06,  7.46it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:06,  7.46it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.05it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.05it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.05it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 10.90it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.16it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.16it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.16it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.16it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.16it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.77it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.90it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:08<00:01, 26.90it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:08<00:01, 26.90it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:08<00:01, 26.90it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:08<00:00, 34.51it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:08<00:00, 34.51it/s]

    Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:08<00:00, 34.51it/s]Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:08<00:00, 34.51it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:08<00:00, 34.51it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:08<00:00, 34.51it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:08<00:00, 34.51it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:08<00:00, 34.51it/s]Compiling num tokens (num_tokens=176):  72%|███████▏  | 42/58 [00:08<00:00, 43.35it/s]Compiling num tokens (num_tokens=160):  72%|███████▏  | 42/58 [00:08<00:00, 43.35it/s]Compiling num tokens (num_tokens=144):  72%|███████▏  | 42/58 [00:08<00:00, 43.35it/s]Compiling num tokens (num_tokens=128):  72%|███████▏  | 42/58 [00:08<00:00, 43.35it/s]Compiling num tokens (num_tokens=112):  72%|███████▏  | 42/58 [00:08<00:00, 43.35it/s]Compiling num tokens (num_tokens=96):  72%|███████▏  | 42/58 [00:08<00:00, 43.35it/s] Compiling num tokens (num_tokens=80):  72%|███████▏  | 42/58 [00:08<00:00, 43.35it/s]

    Compiling num tokens (num_tokens=80):  83%|████████▎ | 48/58 [00:08<00:00, 46.04it/s]Compiling num tokens (num_tokens=64):  83%|████████▎ | 48/58 [00:08<00:00, 46.04it/s]Compiling num tokens (num_tokens=48):  83%|████████▎ | 48/58 [00:08<00:00, 46.04it/s]Compiling num tokens (num_tokens=32):  83%|████████▎ | 48/58 [00:08<00:00, 46.04it/s]Compiling num tokens (num_tokens=28):  83%|████████▎ | 48/58 [00:08<00:00, 46.04it/s]Compiling num tokens (num_tokens=24):  83%|████████▎ | 48/58 [00:08<00:00, 46.04it/s]Compiling num tokens (num_tokens=20):  83%|████████▎ | 48/58 [00:08<00:00, 46.04it/s]Compiling num tokens (num_tokens=16):  83%|████████▎ | 48/58 [00:08<00:00, 46.04it/s]Compiling num tokens (num_tokens=12):  83%|████████▎ | 48/58 [00:08<00:00, 46.04it/s]Compiling num tokens (num_tokens=8):  83%|████████▎ | 48/58 [00:08<00:00, 46.04it/s] Compiling num tokens (num_tokens=8):  98%|█████████▊| 57/58 [00:08<00:00, 57.02it/s]Compiling num tokens (num_tokens=4):  98%|█████████▊| 57/58 [00:08<00:00, 57.02it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  6.92it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=50.08 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=50.08 GB):   2%|▏         | 1/58 [00:00<00:19,  2.86it/s]Capturing num tokens (num_tokens=7680 avail_mem=50.05 GB):   2%|▏         | 1/58 [00:00<00:19,  2.86it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=50.05 GB):   3%|▎         | 2/58 [00:00<00:18,  3.03it/s]Capturing num tokens (num_tokens=7168 avail_mem=50.05 GB):   3%|▎         | 2/58 [00:00<00:18,  3.03it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=50.05 GB):   5%|▌         | 3/58 [00:00<00:16,  3.26it/s]Capturing num tokens (num_tokens=6656 avail_mem=50.05 GB):   5%|▌         | 3/58 [00:00<00:16,  3.26it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=50.05 GB):   7%|▋         | 4/58 [00:01<00:15,  3.44it/s]Capturing num tokens (num_tokens=6144 avail_mem=50.05 GB):   7%|▋         | 4/58 [00:01<00:15,  3.44it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=50.05 GB):   9%|▊         | 5/58 [00:01<00:14,  3.57it/s]Capturing num tokens (num_tokens=5632 avail_mem=50.05 GB):   9%|▊         | 5/58 [00:01<00:14,  3.57it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=50.05 GB):  10%|█         | 6/58 [00:01<00:13,  3.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=50.05 GB):  10%|█         | 6/58 [00:01<00:13,  3.92it/s]Capturing num tokens (num_tokens=5120 avail_mem=50.05 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.30it/s]Capturing num tokens (num_tokens=4608 avail_mem=50.05 GB):  12%|█▏        | 7/58 [00:01<00:11,  4.30it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=50.05 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.05 GB):  14%|█▍        | 8/58 [00:02<00:10,  4.71it/s]Capturing num tokens (num_tokens=4096 avail_mem=50.05 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.15it/s]Capturing num tokens (num_tokens=3840 avail_mem=50.05 GB):  16%|█▌        | 9/58 [00:02<00:09,  5.15it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=50.05 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=50.05 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.60it/s]Capturing num tokens (num_tokens=3584 avail_mem=50.05 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.05it/s]Capturing num tokens (num_tokens=3328 avail_mem=50.04 GB):  19%|█▉        | 11/58 [00:02<00:07,  6.05it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=50.04 GB):  21%|██        | 12/58 [00:02<00:06,  6.61it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.04 GB):  21%|██        | 12/58 [00:02<00:06,  6.61it/s]Capturing num tokens (num_tokens=3072 avail_mem=50.04 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.17it/s]Capturing num tokens (num_tokens=2816 avail_mem=50.04 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.17it/s]

    Capturing num tokens (num_tokens=2816 avail_mem=50.04 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.83it/s]Capturing num tokens (num_tokens=2560 avail_mem=50.04 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=50.04 GB):  24%|██▍       | 14/58 [00:02<00:05,  7.83it/s]Capturing num tokens (num_tokens=2304 avail_mem=50.04 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.22it/s]Capturing num tokens (num_tokens=2048 avail_mem=50.03 GB):  28%|██▊       | 16/58 [00:02<00:04,  9.22it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=50.03 GB):  28%|██▊       | 16/58 [00:03<00:04,  9.22it/s]Capturing num tokens (num_tokens=1792 avail_mem=50.03 GB):  31%|███       | 18/58 [00:03<00:03, 10.93it/s]Capturing num tokens (num_tokens=1536 avail_mem=50.03 GB):  31%|███       | 18/58 [00:03<00:03, 10.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=50.03 GB):  31%|███       | 18/58 [00:03<00:03, 10.93it/s]Capturing num tokens (num_tokens=1280 avail_mem=50.03 GB):  34%|███▍      | 20/58 [00:03<00:02, 12.95it/s]Capturing num tokens (num_tokens=1024 avail_mem=50.02 GB):  34%|███▍      | 20/58 [00:03<00:02, 12.95it/s]

    Capturing num tokens (num_tokens=960 avail_mem=50.01 GB):  34%|███▍      | 20/58 [00:03<00:02, 12.95it/s] Capturing num tokens (num_tokens=896 avail_mem=50.01 GB):  34%|███▍      | 20/58 [00:03<00:02, 12.95it/s]Capturing num tokens (num_tokens=896 avail_mem=50.01 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.09it/s]Capturing num tokens (num_tokens=832 avail_mem=50.00 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.09it/s]Capturing num tokens (num_tokens=768 avail_mem=50.00 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.09it/s]Capturing num tokens (num_tokens=704 avail_mem=49.99 GB):  40%|███▉      | 23/58 [00:03<00:02, 16.09it/s]

    Capturing num tokens (num_tokens=704 avail_mem=49.99 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.52it/s]Capturing num tokens (num_tokens=640 avail_mem=49.99 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.52it/s]Capturing num tokens (num_tokens=576 avail_mem=49.99 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.52it/s]Capturing num tokens (num_tokens=512 avail_mem=49.98 GB):  45%|████▍     | 26/58 [00:03<00:01, 18.52it/s]Capturing num tokens (num_tokens=512 avail_mem=49.98 GB):  50%|█████     | 29/58 [00:03<00:01, 20.93it/s]Capturing num tokens (num_tokens=480 avail_mem=49.98 GB):  50%|█████     | 29/58 [00:03<00:01, 20.93it/s]Capturing num tokens (num_tokens=448 avail_mem=49.97 GB):  50%|█████     | 29/58 [00:03<00:01, 20.93it/s]

    Capturing num tokens (num_tokens=416 avail_mem=49.97 GB):  50%|█████     | 29/58 [00:03<00:01, 20.93it/s]Capturing num tokens (num_tokens=416 avail_mem=49.97 GB):  55%|█████▌    | 32/58 [00:03<00:01, 21.17it/s]Capturing num tokens (num_tokens=384 avail_mem=49.97 GB):  55%|█████▌    | 32/58 [00:03<00:01, 21.17it/s]Capturing num tokens (num_tokens=352 avail_mem=49.97 GB):  55%|█████▌    | 32/58 [00:03<00:01, 21.17it/s]Capturing num tokens (num_tokens=320 avail_mem=49.96 GB):  55%|█████▌    | 32/58 [00:03<00:01, 21.17it/s]Capturing num tokens (num_tokens=320 avail_mem=49.96 GB):  60%|██████    | 35/58 [00:03<00:01, 20.35it/s]Capturing num tokens (num_tokens=288 avail_mem=49.96 GB):  60%|██████    | 35/58 [00:03<00:01, 20.35it/s]

    Capturing num tokens (num_tokens=256 avail_mem=49.95 GB):  60%|██████    | 35/58 [00:03<00:01, 20.35it/s]Capturing num tokens (num_tokens=240 avail_mem=49.95 GB):  60%|██████    | 35/58 [00:03<00:01, 20.35it/s]Capturing num tokens (num_tokens=240 avail_mem=49.95 GB):  66%|██████▌   | 38/58 [00:03<00:00, 22.35it/s]Capturing num tokens (num_tokens=224 avail_mem=49.95 GB):  66%|██████▌   | 38/58 [00:03<00:00, 22.35it/s]Capturing num tokens (num_tokens=208 avail_mem=49.94 GB):  66%|██████▌   | 38/58 [00:04<00:00, 22.35it/s]Capturing num tokens (num_tokens=192 avail_mem=49.94 GB):  66%|██████▌   | 38/58 [00:04<00:00, 22.35it/s]Capturing num tokens (num_tokens=192 avail_mem=49.94 GB):  71%|███████   | 41/58 [00:04<00:00, 23.07it/s]Capturing num tokens (num_tokens=176 avail_mem=49.93 GB):  71%|███████   | 41/58 [00:04<00:00, 23.07it/s]

    Capturing num tokens (num_tokens=160 avail_mem=49.93 GB):  71%|███████   | 41/58 [00:04<00:00, 23.07it/s]Capturing num tokens (num_tokens=144 avail_mem=49.92 GB):  71%|███████   | 41/58 [00:04<00:00, 23.07it/s]Capturing num tokens (num_tokens=144 avail_mem=49.92 GB):  76%|███████▌  | 44/58 [00:04<00:00, 23.45it/s]Capturing num tokens (num_tokens=128 avail_mem=49.92 GB):  76%|███████▌  | 44/58 [00:04<00:00, 23.45it/s]Capturing num tokens (num_tokens=112 avail_mem=49.92 GB):  76%|███████▌  | 44/58 [00:04<00:00, 23.45it/s]Capturing num tokens (num_tokens=96 avail_mem=49.92 GB):  76%|███████▌  | 44/58 [00:04<00:00, 23.45it/s] Capturing num tokens (num_tokens=96 avail_mem=49.92 GB):  81%|████████  | 47/58 [00:04<00:00, 24.32it/s]Capturing num tokens (num_tokens=80 avail_mem=49.91 GB):  81%|████████  | 47/58 [00:04<00:00, 24.32it/s]

    Capturing num tokens (num_tokens=64 avail_mem=49.91 GB):  81%|████████  | 47/58 [00:04<00:00, 24.32it/s]Capturing num tokens (num_tokens=48 avail_mem=49.91 GB):  81%|████████  | 47/58 [00:04<00:00, 24.32it/s]Capturing num tokens (num_tokens=48 avail_mem=49.91 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.18it/s]Capturing num tokens (num_tokens=32 avail_mem=49.90 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.18it/s]Capturing num tokens (num_tokens=28 avail_mem=49.90 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.18it/s]Capturing num tokens (num_tokens=24 avail_mem=49.89 GB):  86%|████████▌ | 50/58 [00:04<00:00, 24.18it/s]

    Capturing num tokens (num_tokens=24 avail_mem=49.89 GB):  91%|█████████▏| 53/58 [00:04<00:00, 22.32it/s]Capturing num tokens (num_tokens=20 avail_mem=49.89 GB):  91%|█████████▏| 53/58 [00:04<00:00, 22.32it/s]Capturing num tokens (num_tokens=16 avail_mem=49.89 GB):  91%|█████████▏| 53/58 [00:04<00:00, 22.32it/s]Capturing num tokens (num_tokens=12 avail_mem=49.88 GB):  91%|█████████▏| 53/58 [00:04<00:00, 22.32it/s]Capturing num tokens (num_tokens=12 avail_mem=49.88 GB):  97%|█████████▋| 56/58 [00:04<00:00, 22.91it/s]Capturing num tokens (num_tokens=8 avail_mem=49.88 GB):  97%|█████████▋| 56/58 [00:04<00:00, 22.91it/s] Capturing num tokens (num_tokens=4 avail_mem=49.87 GB):  97%|█████████▋| 56/58 [00:04<00:00, 22.91it/s]Capturing num tokens (num_tokens=4 avail_mem=49.87 GB): 100%|██████████| 58/58 [00:04<00:00, 12.10it/s]


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


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function><br><br>Sources:<br>- get_current_date function<br>- get_current_weather function</strong>



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


<strong style='color: #00008B;'><function=get_current_date>{"timezone": "America/New_York"}</function><br><function=get_current_weather>{"city": "New York", "state": "NY", "unit": "fahrenheit"}</function><br><br>Sources: <br>- get_current_date function<br>- get_current_weather function</strong>


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


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': '4a004fb5c355429eb089288d2c8f2ca3', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 1, 'cached_tokens_details': {'device': 1, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1543846782296896, 'response_sent_to_client_ts': 1780952262.3775113}}</strong>



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


<strong style='color: #00008B;'>{'text': '{"name": "Paris", "population": 2147000}', 'output_ids': [5018, 609, 794, 330, 60704, 498, 330, 45541, 794, 220, 11584, 7007, 15, 92, 128009], 'meta_info': {'id': '9ceaf4db9de74d0781972462086d2c43', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 50, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 15, 'cached_tokens': 49, 'cached_tokens_details': {'device': 49, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.1510826889425516, 'response_sent_to_client_ts': 1780952262.5429642}}</strong>


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


<strong style='color: #00008B;'>[{'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '45653ef15e94405ca73a79e1ef781e09', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.5188383460044861, 'response_sent_to_client_ts': 1780952263.0852487}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': 'da7821c69cec45bfb19353eb07885c2b', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.5187648609280586, 'response_sent_to_client_ts': 1780952263.0852625}}, {'text': 'Paris is the capital of France', 'output_ids': [60704, 374, 279, 6864, 315, 9822, 128009], 'meta_info': {'id': '0e00f57e450644c6a31bd1cb7317ecf0', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 46, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 7, 'cached_tokens': 45, 'cached_tokens_details': {'device': 45, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.5187211539596319, 'response_sent_to_client_ts': 1780952263.0852666}}]</strong>


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


<strong style='color: #00008B;'>{'text': 'France', 'output_ids': [50100, 128009], 'meta_info': {'id': 'eaccf5e93f9f41d79c456196f43a41df', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 2, 'cached_tokens': 31, 'cached_tokens_details': {'device': 31, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.03228250332176685, 'response_sent_to_client_ts': 1780952263.1254506}}</strong>


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


<strong style='color: #00008B;'>{'text': 'France.', 'output_ids': [50100, 13, 128009], 'meta_info': {'id': '669deea55acb41eaaa31c2c83d91a7f4', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 3, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.11438241600990295, 'response_sent_to_client_ts': 1780952264.5438845}}</strong>



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


<strong style='color: #00008B;'>{'text': 'France.', 'output_ids': [50100, 13, 128009], 'meta_info': {'id': '2233c86ece5e4bb2b4e8048c8dd44a65', 'finish_reason': {'type': 'stop', 'matched': 128009}, 'prompt_tokens': 41, 'weight_version': 'default', 'num_retractions': 0, 'reasoning_tokens': 0, 'completion_tokens': 3, 'cached_tokens': 40, 'cached_tokens_details': {'device': 40, 'host': 0}, 'dp_rank': None, 'e2e_latency': 0.041942035779356956, 'response_sent_to_client_ts': 1780952264.5977347}}</strong>



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

    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:00<00:02,  1.23it/s]

    Multi-thread loading shards:  50% Completed | 2/4 [00:01<00:01,  1.13it/s]

    Multi-thread loading shards:  75% Completed | 3/4 [00:02<00:00,  1.10it/s]

    Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.47it/s]Multi-thread loading shards: 100% Completed | 4/4 [00:03<00:00,  1.32it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:09,  5.44s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:09,  5.44s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:13,  2.38s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:13,  2.38s/it]

    Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:16,  1.39s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:16,  1.39s/it]Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:49,  1.09it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:49,  1.09it/s]

    Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:34,  1.52it/s]Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:25,  2.04it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:25,  2.04it/s]

    Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:19,  2.65it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:19,  2.65it/s]Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:15,  3.33it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:15,  3.33it/s]

    Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.09it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.09it/s]Compiling num tokens (num_tokens=3840):  17%|█▋        | 10/58 [00:06<00:09,  4.96it/s]Compiling num tokens (num_tokens=3584):  17%|█▋        | 10/58 [00:06<00:09,  4.96it/s]

    Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:08,  5.85it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:08,  5.85it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:08,  5.85it/s]Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:05,  7.52it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:05,  7.52it/s]

    Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:05,  7.52it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  8.93it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  8.93it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  8.93it/s]Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 10.84it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 10.84it/s]

    Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 10.84it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 10.84it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.09it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.09it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.09it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.09it/s]Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.09it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.68it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.68it/s]

    Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.68it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.68it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.68it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.68it/s]Compiling num tokens (num_tokens=512):  50%|█████     | 29/58 [00:07<00:01, 26.96it/s]Compiling num tokens (num_tokens=480):  50%|█████     | 29/58 [00:07<00:01, 26.96it/s]Compiling num tokens (num_tokens=448):  50%|█████     | 29/58 [00:07<00:01, 26.96it/s]Compiling num tokens (num_tokens=416):  50%|█████     | 29/58 [00:07<00:01, 26.96it/s]Compiling num tokens (num_tokens=384):  50%|█████     | 29/58 [00:07<00:01, 26.96it/s]Compiling num tokens (num_tokens=352):  50%|█████     | 29/58 [00:07<00:01, 26.96it/s]Compiling num tokens (num_tokens=320):  50%|█████     | 29/58 [00:07<00:01, 26.96it/s]Compiling num tokens (num_tokens=320):  60%|██████    | 35/58 [00:07<00:00, 34.78it/s]Compiling num tokens (num_tokens=288):  60%|██████    | 35/58 [00:07<00:00, 34.78it/s]Compiling num tokens (num_tokens=256):  60%|██████    | 35/58 [00:07<00:00, 34.78it/s]

    Compiling num tokens (num_tokens=240):  60%|██████    | 35/58 [00:07<00:00, 34.78it/s]Compiling num tokens (num_tokens=224):  60%|██████    | 35/58 [00:07<00:00, 34.78it/s]Compiling num tokens (num_tokens=208):  60%|██████    | 35/58 [00:08<00:00, 34.78it/s]Compiling num tokens (num_tokens=192):  60%|██████    | 35/58 [00:08<00:00, 34.78it/s]Compiling num tokens (num_tokens=176):  60%|██████    | 35/58 [00:08<00:00, 34.78it/s]Compiling num tokens (num_tokens=160):  60%|██████    | 35/58 [00:08<00:00, 34.78it/s]Compiling num tokens (num_tokens=160):  74%|███████▍  | 43/58 [00:08<00:00, 44.96it/s]Compiling num tokens (num_tokens=144):  74%|███████▍  | 43/58 [00:08<00:00, 44.96it/s]Compiling num tokens (num_tokens=128):  74%|███████▍  | 43/58 [00:08<00:00, 44.96it/s]Compiling num tokens (num_tokens=112):  74%|███████▍  | 43/58 [00:08<00:00, 44.96it/s]Compiling num tokens (num_tokens=96):  74%|███████▍  | 43/58 [00:08<00:00, 44.96it/s] Compiling num tokens (num_tokens=80):  74%|███████▍  | 43/58 [00:08<00:00, 44.96it/s]Compiling num tokens (num_tokens=64):  74%|███████▍  | 43/58 [00:08<00:00, 44.96it/s]Compiling num tokens (num_tokens=48):  74%|███████▍  | 43/58 [00:08<00:00, 44.96it/s]

    Compiling num tokens (num_tokens=48):  86%|████████▌ | 50/58 [00:08<00:00, 49.95it/s]Compiling num tokens (num_tokens=32):  86%|████████▌ | 50/58 [00:08<00:00, 49.95it/s]Compiling num tokens (num_tokens=28):  86%|████████▌ | 50/58 [00:08<00:00, 49.95it/s]Compiling num tokens (num_tokens=24):  86%|████████▌ | 50/58 [00:08<00:00, 49.95it/s]Compiling num tokens (num_tokens=20):  86%|████████▌ | 50/58 [00:08<00:00, 49.95it/s]Compiling num tokens (num_tokens=16):  86%|████████▌ | 50/58 [00:08<00:00, 49.95it/s]Compiling num tokens (num_tokens=12):  86%|████████▌ | 50/58 [00:08<00:00, 49.95it/s]Compiling num tokens (num_tokens=8):  86%|████████▌ | 50/58 [00:08<00:00, 49.95it/s] Compiling num tokens (num_tokens=4):  86%|████████▌ | 50/58 [00:08<00:00, 49.95it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.01it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=43.36 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=43.36 GB):   2%|▏         | 1/58 [00:00<00:19,  2.88it/s]Capturing num tokens (num_tokens=7680 avail_mem=43.33 GB):   2%|▏         | 1/58 [00:00<00:19,  2.88it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=43.33 GB):   3%|▎         | 2/58 [00:00<00:18,  3.05it/s]Capturing num tokens (num_tokens=7168 avail_mem=43.33 GB):   3%|▎         | 2/58 [00:00<00:18,  3.05it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=43.33 GB):   5%|▌         | 3/58 [00:00<00:16,  3.30it/s]Capturing num tokens (num_tokens=6656 avail_mem=43.33 GB):   5%|▌         | 3/58 [00:00<00:16,  3.30it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=43.33 GB):   7%|▋         | 4/58 [00:01<00:19,  2.83it/s]Capturing num tokens (num_tokens=6144 avail_mem=42.19 GB):   7%|▋         | 4/58 [00:01<00:19,  2.83it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=42.19 GB):   9%|▊         | 5/58 [00:01<00:18,  2.94it/s]Capturing num tokens (num_tokens=5632 avail_mem=41.55 GB):   9%|▊         | 5/58 [00:01<00:18,  2.94it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=41.55 GB):  10%|█         | 6/58 [00:01<00:15,  3.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.56 GB):  10%|█         | 6/58 [00:01<00:15,  3.41it/s]Capturing num tokens (num_tokens=5120 avail_mem=41.56 GB):  12%|█▏        | 7/58 [00:02<00:13,  3.87it/s]Capturing num tokens (num_tokens=4608 avail_mem=41.55 GB):  12%|█▏        | 7/58 [00:02<00:13,  3.87it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=41.55 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.55 GB):  14%|█▍        | 8/58 [00:02<00:11,  4.37it/s]Capturing num tokens (num_tokens=4096 avail_mem=41.55 GB):  16%|█▌        | 9/58 [00:02<00:09,  4.93it/s]Capturing num tokens (num_tokens=3840 avail_mem=41.55 GB):  16%|█▌        | 9/58 [00:02<00:09,  4.93it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=41.55 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.55 GB):  17%|█▋        | 10/58 [00:02<00:08,  5.46it/s]Capturing num tokens (num_tokens=3584 avail_mem=41.55 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.98it/s]Capturing num tokens (num_tokens=3328 avail_mem=41.55 GB):  19%|█▉        | 11/58 [00:02<00:07,  5.98it/s]

    Capturing num tokens (num_tokens=3328 avail_mem=41.55 GB):  21%|██        | 12/58 [00:02<00:07,  6.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.55 GB):  21%|██        | 12/58 [00:02<00:07,  6.57it/s]Capturing num tokens (num_tokens=3072 avail_mem=41.55 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.20it/s]Capturing num tokens (num_tokens=2816 avail_mem=41.54 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.20it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=41.54 GB):  22%|██▏       | 13/58 [00:02<00:06,  7.20it/s]Capturing num tokens (num_tokens=2560 avail_mem=41.54 GB):  26%|██▌       | 15/58 [00:03<00:05,  8.56it/s]Capturing num tokens (num_tokens=2304 avail_mem=41.54 GB):  26%|██▌       | 15/58 [00:03<00:05,  8.56it/s]Capturing num tokens (num_tokens=2048 avail_mem=41.54 GB):  26%|██▌       | 15/58 [00:03<00:05,  8.56it/s]

    Capturing num tokens (num_tokens=2048 avail_mem=41.54 GB):  29%|██▉       | 17/58 [00:03<00:04,  9.99it/s]Capturing num tokens (num_tokens=1792 avail_mem=41.53 GB):  29%|██▉       | 17/58 [00:03<00:04,  9.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.53 GB):  29%|██▉       | 17/58 [00:03<00:04,  9.99it/s]Capturing num tokens (num_tokens=1536 avail_mem=41.53 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.80it/s]Capturing num tokens (num_tokens=1280 avail_mem=41.53 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.80it/s]Capturing num tokens (num_tokens=1024 avail_mem=41.53 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.80it/s]

    Capturing num tokens (num_tokens=960 avail_mem=41.51 GB):  33%|███▎      | 19/58 [00:03<00:03, 11.80it/s] Capturing num tokens (num_tokens=960 avail_mem=41.51 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.98it/s]Capturing num tokens (num_tokens=896 avail_mem=41.51 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.98it/s]Capturing num tokens (num_tokens=832 avail_mem=41.51 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.98it/s]Capturing num tokens (num_tokens=768 avail_mem=41.50 GB):  38%|███▊      | 22/58 [00:03<00:02, 14.98it/s]Capturing num tokens (num_tokens=768 avail_mem=41.50 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.68it/s]Capturing num tokens (num_tokens=704 avail_mem=41.50 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.68it/s]Capturing num tokens (num_tokens=640 avail_mem=41.49 GB):  43%|████▎     | 25/58 [00:03<00:01, 17.68it/s]

    Capturing num tokens (num_tokens=640 avail_mem=41.49 GB):  47%|████▋     | 27/58 [00:03<00:01, 15.61it/s]Capturing num tokens (num_tokens=576 avail_mem=41.49 GB):  47%|████▋     | 27/58 [00:03<00:01, 15.61it/s]Capturing num tokens (num_tokens=512 avail_mem=41.48 GB):  47%|████▋     | 27/58 [00:03<00:01, 15.61it/s]

    Capturing num tokens (num_tokens=512 avail_mem=41.48 GB):  50%|█████     | 29/58 [00:03<00:02, 12.54it/s]Capturing num tokens (num_tokens=480 avail_mem=41.48 GB):  50%|█████     | 29/58 [00:03<00:02, 12.54it/s]Capturing num tokens (num_tokens=448 avail_mem=41.48 GB):  50%|█████     | 29/58 [00:04<00:02, 12.54it/s]

    Capturing num tokens (num_tokens=448 avail_mem=41.48 GB):  53%|█████▎    | 31/58 [00:04<00:02, 10.95it/s]Capturing num tokens (num_tokens=416 avail_mem=41.48 GB):  53%|█████▎    | 31/58 [00:04<00:02, 10.95it/s]Capturing num tokens (num_tokens=384 avail_mem=41.47 GB):  53%|█████▎    | 31/58 [00:04<00:02, 10.95it/s]

    Capturing num tokens (num_tokens=384 avail_mem=41.47 GB):  57%|█████▋    | 33/58 [00:04<00:02, 10.06it/s]Capturing num tokens (num_tokens=352 avail_mem=40.41 GB):  57%|█████▋    | 33/58 [00:04<00:02, 10.06it/s]Capturing num tokens (num_tokens=320 avail_mem=26.45 GB):  57%|█████▋    | 33/58 [00:04<00:02, 10.06it/s]

    Capturing num tokens (num_tokens=320 avail_mem=26.45 GB):  60%|██████    | 35/58 [00:04<00:02,  9.51it/s]Capturing num tokens (num_tokens=288 avail_mem=26.44 GB):  60%|██████    | 35/58 [00:04<00:02,  9.51it/s]Capturing num tokens (num_tokens=256 avail_mem=26.44 GB):  60%|██████    | 35/58 [00:04<00:02,  9.51it/s]

    Capturing num tokens (num_tokens=256 avail_mem=26.44 GB):  64%|██████▍   | 37/58 [00:04<00:02,  9.25it/s]Capturing num tokens (num_tokens=240 avail_mem=26.43 GB):  64%|██████▍   | 37/58 [00:04<00:02,  9.25it/s]Capturing num tokens (num_tokens=240 avail_mem=26.43 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.27it/s]Capturing num tokens (num_tokens=224 avail_mem=26.43 GB):  66%|██████▌   | 38/58 [00:05<00:02,  9.27it/s]

    Capturing num tokens (num_tokens=224 avail_mem=26.43 GB):  67%|██████▋   | 39/58 [00:05<00:02,  9.11it/s]Capturing num tokens (num_tokens=208 avail_mem=26.42 GB):  67%|██████▋   | 39/58 [00:05<00:02,  9.11it/s]Capturing num tokens (num_tokens=208 avail_mem=26.42 GB):  69%|██████▉   | 40/58 [00:05<00:01,  9.15it/s]Capturing num tokens (num_tokens=192 avail_mem=26.42 GB):  69%|██████▉   | 40/58 [00:05<00:01,  9.15it/s]

    Capturing num tokens (num_tokens=192 avail_mem=26.42 GB):  71%|███████   | 41/58 [00:05<00:01,  9.22it/s]Capturing num tokens (num_tokens=176 avail_mem=26.41 GB):  71%|███████   | 41/58 [00:05<00:01,  9.22it/s]Capturing num tokens (num_tokens=176 avail_mem=26.41 GB):  72%|███████▏  | 42/58 [00:05<00:01,  9.28it/s]Capturing num tokens (num_tokens=160 avail_mem=26.41 GB):  72%|███████▏  | 42/58 [00:05<00:01,  9.28it/s]

    Capturing num tokens (num_tokens=160 avail_mem=26.41 GB):  74%|███████▍  | 43/58 [00:05<00:01,  9.35it/s]Capturing num tokens (num_tokens=144 avail_mem=26.40 GB):  74%|███████▍  | 43/58 [00:05<00:01,  9.35it/s]Capturing num tokens (num_tokens=144 avail_mem=26.40 GB):  76%|███████▌  | 44/58 [00:05<00:01,  9.38it/s]Capturing num tokens (num_tokens=128 avail_mem=26.40 GB):  76%|███████▌  | 44/58 [00:05<00:01,  9.38it/s]

    Capturing num tokens (num_tokens=112 avail_mem=26.40 GB):  76%|███████▌  | 44/58 [00:05<00:01,  9.38it/s]Capturing num tokens (num_tokens=112 avail_mem=26.40 GB):  79%|███████▉  | 46/58 [00:05<00:01,  9.63it/s]Capturing num tokens (num_tokens=96 avail_mem=26.40 GB):  79%|███████▉  | 46/58 [00:05<00:01,  9.63it/s] 

    Capturing num tokens (num_tokens=80 avail_mem=26.40 GB):  79%|███████▉  | 46/58 [00:05<00:01,  9.63it/s]Capturing num tokens (num_tokens=80 avail_mem=26.40 GB):  83%|████████▎ | 48/58 [00:06<00:01,  9.85it/s]Capturing num tokens (num_tokens=64 avail_mem=26.39 GB):  83%|████████▎ | 48/58 [00:06<00:01,  9.85it/s]Capturing num tokens (num_tokens=48 avail_mem=26.39 GB):  83%|████████▎ | 48/58 [00:06<00:01,  9.85it/s]

    Capturing num tokens (num_tokens=48 avail_mem=26.39 GB):  86%|████████▌ | 50/58 [00:06<00:00, 10.38it/s]Capturing num tokens (num_tokens=32 avail_mem=26.39 GB):  86%|████████▌ | 50/58 [00:06<00:00, 10.38it/s]Capturing num tokens (num_tokens=28 avail_mem=26.38 GB):  86%|████████▌ | 50/58 [00:06<00:00, 10.38it/s]Capturing num tokens (num_tokens=28 avail_mem=26.38 GB):  90%|████████▉ | 52/58 [00:06<00:00, 11.19it/s]Capturing num tokens (num_tokens=24 avail_mem=26.38 GB):  90%|████████▉ | 52/58 [00:06<00:00, 11.19it/s]

    Capturing num tokens (num_tokens=20 avail_mem=26.37 GB):  90%|████████▉ | 52/58 [00:06<00:00, 11.19it/s]Capturing num tokens (num_tokens=20 avail_mem=26.37 GB):  93%|█████████▎| 54/58 [00:06<00:00, 10.85it/s]Capturing num tokens (num_tokens=16 avail_mem=26.37 GB):  93%|█████████▎| 54/58 [00:06<00:00, 10.85it/s]Capturing num tokens (num_tokens=12 avail_mem=26.36 GB):  93%|█████████▎| 54/58 [00:06<00:00, 10.85it/s]Capturing num tokens (num_tokens=12 avail_mem=26.36 GB):  97%|█████████▋| 56/58 [00:06<00:00, 12.49it/s]Capturing num tokens (num_tokens=8 avail_mem=26.36 GB):  97%|█████████▋| 56/58 [00:06<00:00, 12.49it/s] 

    Capturing num tokens (num_tokens=4 avail_mem=26.36 GB):  97%|█████████▋| 56/58 [00:06<00:00, 12.49it/s]Capturing num tokens (num_tokens=4 avail_mem=26.36 GB): 100%|██████████| 58/58 [00:06<00:00,  8.52it/s]


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



<strong style='color: #00008B;'>Prompt: Give me the information of the capital of Italy.<br>Generated text: Paris is the capital of France</strong>


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
