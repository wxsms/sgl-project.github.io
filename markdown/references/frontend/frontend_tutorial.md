# SGLang Frontend Language

SGLang frontend language can be used to define simple and easy prompts in a convenient, structured way.

## Launch A Server

Launch the server in your terminal and wait for it to initialize.


```python
from sglang import assistant_begin, assistant_end
from sglang import assistant, function, gen, system, user
from sglang import image
from sglang import RuntimeEndpoint
from sglang.lang.api import set_default_backend
from sglang.srt.utils import load_image
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import print_highlight, terminate_process, wait_for_server

server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    Multi-thread loading shards:   0% Completed | 0/4 [00:00<?, ?it/s]

    Multi-thread loading shards:  25% Completed | 1/4 [00:01<00:05,  1.67s/it]

    Multi-thread loading shards:  50% Completed | 2/4 [00:03<00:02,  1.48s/it]

    Multi-thread loading shards:  75% Completed | 3/4 [00:03<00:01,  1.22s/it]

    Multi-thread loading shards: 100% Completed | 4/4 [00:04<00:00,  1.02s/it]Multi-thread loading shards: 100% Completed | 4/4 [00:04<00:00,  1.16s/it]


    2026-05-21 00:32:52,067 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-21 00:32:52] Unexpected error during package walk: cutlass.cute.experimental


      0%|          | 0/58 [00:00<?, ?it/s]Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]

    Compiling num tokens (num_tokens=8192):   2%|▏         | 1/58 [00:05<05:16,  5.55s/it]Compiling num tokens (num_tokens=7680):   2%|▏         | 1/58 [00:05<05:16,  5.55s/it]

    Compiling num tokens (num_tokens=7680):   3%|▎         | 2/58 [00:05<02:15,  2.41s/it]Compiling num tokens (num_tokens=7168):   3%|▎         | 2/58 [00:05<02:15,  2.41s/it]Compiling num tokens (num_tokens=7168):   5%|▌         | 3/58 [00:05<01:17,  1.40s/it]Compiling num tokens (num_tokens=6656):   5%|▌         | 3/58 [00:05<01:17,  1.40s/it]

    Compiling num tokens (num_tokens=6656):   7%|▋         | 4/58 [00:06<00:49,  1.08it/s]Compiling num tokens (num_tokens=6144):   7%|▋         | 4/58 [00:06<00:49,  1.08it/s]Compiling num tokens (num_tokens=6144):   9%|▊         | 5/58 [00:06<00:34,  1.55it/s]Compiling num tokens (num_tokens=5632):   9%|▊         | 5/58 [00:06<00:34,  1.55it/s]

    Compiling num tokens (num_tokens=5632):  10%|█         | 6/58 [00:06<00:24,  2.09it/s]Compiling num tokens (num_tokens=5120):  10%|█         | 6/58 [00:06<00:24,  2.09it/s]Compiling num tokens (num_tokens=5120):  12%|█▏        | 7/58 [00:06<00:18,  2.73it/s]Compiling num tokens (num_tokens=4608):  12%|█▏        | 7/58 [00:06<00:18,  2.73it/s]

    Compiling num tokens (num_tokens=4608):  14%|█▍        | 8/58 [00:06<00:14,  3.47it/s]Compiling num tokens (num_tokens=4096):  14%|█▍        | 8/58 [00:06<00:14,  3.47it/s]Compiling num tokens (num_tokens=4096):  16%|█▌        | 9/58 [00:06<00:11,  4.33it/s]Compiling num tokens (num_tokens=3840):  16%|█▌        | 9/58 [00:06<00:11,  4.33it/s]

    Compiling num tokens (num_tokens=3584):  16%|█▌        | 9/58 [00:06<00:11,  4.33it/s]Compiling num tokens (num_tokens=3584):  19%|█▉        | 11/58 [00:07<00:07,  6.02it/s]Compiling num tokens (num_tokens=3328):  19%|█▉        | 11/58 [00:07<00:07,  6.02it/s]Compiling num tokens (num_tokens=3072):  19%|█▉        | 11/58 [00:07<00:07,  6.02it/s]

    Compiling num tokens (num_tokens=3072):  22%|██▏       | 13/58 [00:07<00:05,  7.60it/s]Compiling num tokens (num_tokens=2816):  22%|██▏       | 13/58 [00:07<00:05,  7.60it/s]Compiling num tokens (num_tokens=2560):  22%|██▏       | 13/58 [00:07<00:05,  7.60it/s]Compiling num tokens (num_tokens=2560):  26%|██▌       | 15/58 [00:07<00:04,  9.22it/s]Compiling num tokens (num_tokens=2304):  26%|██▌       | 15/58 [00:07<00:04,  9.22it/s]Compiling num tokens (num_tokens=2048):  26%|██▌       | 15/58 [00:07<00:04,  9.22it/s]

    Compiling num tokens (num_tokens=2048):  29%|██▉       | 17/58 [00:07<00:03, 11.16it/s]Compiling num tokens (num_tokens=1792):  29%|██▉       | 17/58 [00:07<00:03, 11.16it/s]Compiling num tokens (num_tokens=1536):  29%|██▉       | 17/58 [00:07<00:03, 11.16it/s]Compiling num tokens (num_tokens=1280):  29%|██▉       | 17/58 [00:07<00:03, 11.16it/s]Compiling num tokens (num_tokens=1280):  34%|███▍      | 20/58 [00:07<00:02, 14.41it/s]Compiling num tokens (num_tokens=1024):  34%|███▍      | 20/58 [00:07<00:02, 14.41it/s]Compiling num tokens (num_tokens=960):  34%|███▍      | 20/58 [00:07<00:02, 14.41it/s] Compiling num tokens (num_tokens=896):  34%|███▍      | 20/58 [00:07<00:02, 14.41it/s]

    Compiling num tokens (num_tokens=832):  34%|███▍      | 20/58 [00:07<00:02, 14.41it/s]Compiling num tokens (num_tokens=832):  41%|████▏     | 24/58 [00:07<00:01, 19.84it/s]Compiling num tokens (num_tokens=768):  41%|████▏     | 24/58 [00:07<00:01, 19.84it/s]Compiling num tokens (num_tokens=704):  41%|████▏     | 24/58 [00:07<00:01, 19.84it/s]Compiling num tokens (num_tokens=640):  41%|████▏     | 24/58 [00:07<00:01, 19.84it/s]Compiling num tokens (num_tokens=576):  41%|████▏     | 24/58 [00:07<00:01, 19.84it/s]Compiling num tokens (num_tokens=512):  41%|████▏     | 24/58 [00:07<00:01, 19.84it/s]Compiling num tokens (num_tokens=480):  41%|████▏     | 24/58 [00:07<00:01, 19.84it/s]Compiling num tokens (num_tokens=480):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=448):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=416):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=384):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=352):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]

    Compiling num tokens (num_tokens=320):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=288):  52%|█████▏    | 30/58 [00:07<00:00, 28.68it/s]Compiling num tokens (num_tokens=288):  62%|██████▏   | 36/58 [00:07<00:00, 36.05it/s]Compiling num tokens (num_tokens=256):  62%|██████▏   | 36/58 [00:07<00:00, 36.05it/s]Compiling num tokens (num_tokens=240):  62%|██████▏   | 36/58 [00:07<00:00, 36.05it/s]Compiling num tokens (num_tokens=224):  62%|██████▏   | 36/58 [00:07<00:00, 36.05it/s]Compiling num tokens (num_tokens=208):  62%|██████▏   | 36/58 [00:07<00:00, 36.05it/s]Compiling num tokens (num_tokens=192):  62%|██████▏   | 36/58 [00:07<00:00, 36.05it/s]Compiling num tokens (num_tokens=176):  62%|██████▏   | 36/58 [00:07<00:00, 36.05it/s]Compiling num tokens (num_tokens=160):  62%|██████▏   | 36/58 [00:07<00:00, 36.05it/s]Compiling num tokens (num_tokens=144):  62%|██████▏   | 36/58 [00:07<00:00, 36.05it/s]Compiling num tokens (num_tokens=144):  76%|███████▌  | 44/58 [00:07<00:00, 45.87it/s]Compiling num tokens (num_tokens=128):  76%|███████▌  | 44/58 [00:07<00:00, 45.87it/s]Compiling num tokens (num_tokens=112):  76%|███████▌  | 44/58 [00:07<00:00, 45.87it/s]Compiling num tokens (num_tokens=96):  76%|███████▌  | 44/58 [00:08<00:00, 45.87it/s] Compiling num tokens (num_tokens=80):  76%|███████▌  | 44/58 [00:08<00:00, 45.87it/s]

    Compiling num tokens (num_tokens=64):  76%|███████▌  | 44/58 [00:08<00:00, 45.87it/s]Compiling num tokens (num_tokens=48):  76%|███████▌  | 44/58 [00:08<00:00, 45.87it/s]Compiling num tokens (num_tokens=32):  76%|███████▌  | 44/58 [00:08<00:00, 45.87it/s]Compiling num tokens (num_tokens=32):  88%|████████▊ | 51/58 [00:08<00:00, 52.24it/s]Compiling num tokens (num_tokens=28):  88%|████████▊ | 51/58 [00:08<00:00, 52.24it/s]Compiling num tokens (num_tokens=24):  88%|████████▊ | 51/58 [00:08<00:00, 52.24it/s]Compiling num tokens (num_tokens=20):  88%|████████▊ | 51/58 [00:08<00:00, 52.24it/s]Compiling num tokens (num_tokens=16):  88%|████████▊ | 51/58 [00:08<00:00, 52.24it/s]Compiling num tokens (num_tokens=12):  88%|████████▊ | 51/58 [00:08<00:00, 52.24it/s]Compiling num tokens (num_tokens=8):  88%|████████▊ | 51/58 [00:08<00:00, 52.24it/s] Compiling num tokens (num_tokens=4):  88%|████████▊ | 51/58 [00:08<00:00, 52.24it/s]Compiling num tokens (num_tokens=4): 100%|██████████| 58/58 [00:08<00:00,  7.11it/s]


      0%|          | 0/58 [00:00<?, ?it/s]Capturing num tokens (num_tokens=8192 avail_mem=38.64 GB):   0%|          | 0/58 [00:00<?, ?it/s]

    Capturing num tokens (num_tokens=8192 avail_mem=38.64 GB):   2%|▏         | 1/58 [00:00<00:29,  1.92it/s]Capturing num tokens (num_tokens=7680 avail_mem=38.60 GB):   2%|▏         | 1/58 [00:00<00:29,  1.92it/s]

    Capturing num tokens (num_tokens=7680 avail_mem=38.60 GB):   3%|▎         | 2/58 [00:01<00:30,  1.82it/s]Capturing num tokens (num_tokens=7168 avail_mem=38.60 GB):   3%|▎         | 2/58 [00:01<00:30,  1.82it/s]

    Capturing num tokens (num_tokens=7168 avail_mem=38.60 GB):   5%|▌         | 3/58 [00:01<00:29,  1.85it/s]Capturing num tokens (num_tokens=6656 avail_mem=38.60 GB):   5%|▌         | 3/58 [00:01<00:29,  1.85it/s]

    Capturing num tokens (num_tokens=6656 avail_mem=38.60 GB):   7%|▋         | 4/58 [00:02<00:27,  1.95it/s]Capturing num tokens (num_tokens=6144 avail_mem=44.35 GB):   7%|▋         | 4/58 [00:02<00:27,  1.95it/s]

    Capturing num tokens (num_tokens=6144 avail_mem=44.35 GB):   9%|▊         | 5/58 [00:02<00:26,  1.96it/s]Capturing num tokens (num_tokens=5632 avail_mem=44.35 GB):   9%|▊         | 5/58 [00:02<00:26,  1.96it/s]

    Capturing num tokens (num_tokens=5632 avail_mem=44.35 GB):  10%|█         | 6/58 [00:03<00:24,  2.11it/s]Capturing num tokens (num_tokens=5120 avail_mem=44.35 GB):  10%|█         | 6/58 [00:03<00:24,  2.11it/s]

    Capturing num tokens (num_tokens=5120 avail_mem=44.35 GB):  12%|█▏        | 7/58 [00:03<00:22,  2.28it/s]Capturing num tokens (num_tokens=4608 avail_mem=44.35 GB):  12%|█▏        | 7/58 [00:03<00:22,  2.28it/s]

    Capturing num tokens (num_tokens=4608 avail_mem=44.35 GB):  14%|█▍        | 8/58 [00:03<00:20,  2.49it/s]Capturing num tokens (num_tokens=4096 avail_mem=44.35 GB):  14%|█▍        | 8/58 [00:03<00:20,  2.49it/s]

    Capturing num tokens (num_tokens=4096 avail_mem=44.35 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.74it/s]Capturing num tokens (num_tokens=3840 avail_mem=44.35 GB):  16%|█▌        | 9/58 [00:03<00:17,  2.74it/s]

    Capturing num tokens (num_tokens=3840 avail_mem=44.35 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.01it/s]Capturing num tokens (num_tokens=3584 avail_mem=44.35 GB):  17%|█▋        | 10/58 [00:04<00:15,  3.01it/s]

    Capturing num tokens (num_tokens=3584 avail_mem=44.35 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.35 GB):  19%|█▉        | 11/58 [00:04<00:14,  3.35it/s]Capturing num tokens (num_tokens=3328 avail_mem=44.35 GB):  21%|██        | 12/58 [00:04<00:12,  3.76it/s]Capturing num tokens (num_tokens=3072 avail_mem=44.34 GB):  21%|██        | 12/58 [00:04<00:12,  3.76it/s]

    Capturing num tokens (num_tokens=3072 avail_mem=44.34 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=44.34 GB):  22%|██▏       | 13/58 [00:04<00:10,  4.30it/s]Capturing num tokens (num_tokens=2816 avail_mem=44.34 GB):  24%|██▍       | 14/58 [00:04<00:08,  4.90it/s]Capturing num tokens (num_tokens=2560 avail_mem=44.34 GB):  24%|██▍       | 14/58 [00:04<00:08,  4.90it/s]

    Capturing num tokens (num_tokens=2560 avail_mem=44.34 GB):  26%|██▌       | 15/58 [00:05<00:07,  5.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.34 GB):  26%|██▌       | 15/58 [00:05<00:07,  5.59it/s]Capturing num tokens (num_tokens=2304 avail_mem=44.34 GB):  28%|██▊       | 16/58 [00:05<00:06,  6.42it/s]Capturing num tokens (num_tokens=2048 avail_mem=44.33 GB):  28%|██▊       | 16/58 [00:05<00:06,  6.42it/s]

    Capturing num tokens (num_tokens=1792 avail_mem=44.33 GB):  28%|██▊       | 16/58 [00:05<00:06,  6.42it/s]Capturing num tokens (num_tokens=1792 avail_mem=44.33 GB):  31%|███       | 18/58 [00:05<00:05,  7.58it/s]Capturing num tokens (num_tokens=1536 avail_mem=44.33 GB):  31%|███       | 18/58 [00:05<00:05,  7.58it/s]Capturing num tokens (num_tokens=1280 avail_mem=44.33 GB):  31%|███       | 18/58 [00:05<00:05,  7.58it/s]

    Capturing num tokens (num_tokens=1280 avail_mem=44.33 GB):  34%|███▍      | 20/58 [00:05<00:03, 10.02it/s]Capturing num tokens (num_tokens=1024 avail_mem=44.31 GB):  34%|███▍      | 20/58 [00:05<00:03, 10.02it/s]Capturing num tokens (num_tokens=960 avail_mem=44.31 GB):  34%|███▍      | 20/58 [00:05<00:03, 10.02it/s] Capturing num tokens (num_tokens=896 avail_mem=44.31 GB):  34%|███▍      | 20/58 [00:05<00:03, 10.02it/s]Capturing num tokens (num_tokens=896 avail_mem=44.31 GB):  40%|███▉      | 23/58 [00:05<00:02, 13.87it/s]Capturing num tokens (num_tokens=832 avail_mem=44.30 GB):  40%|███▉      | 23/58 [00:05<00:02, 13.87it/s]Capturing num tokens (num_tokens=768 avail_mem=44.30 GB):  40%|███▉      | 23/58 [00:05<00:02, 13.87it/s]Capturing num tokens (num_tokens=704 avail_mem=44.29 GB):  40%|███▉      | 23/58 [00:05<00:02, 13.87it/s]

    Capturing num tokens (num_tokens=704 avail_mem=44.29 GB):  45%|████▍     | 26/58 [00:05<00:01, 17.36it/s]Capturing num tokens (num_tokens=640 avail_mem=44.29 GB):  45%|████▍     | 26/58 [00:05<00:01, 17.36it/s]Capturing num tokens (num_tokens=576 avail_mem=44.29 GB):  45%|████▍     | 26/58 [00:05<00:01, 17.36it/s]Capturing num tokens (num_tokens=512 avail_mem=44.28 GB):  45%|████▍     | 26/58 [00:05<00:01, 17.36it/s]Capturing num tokens (num_tokens=480 avail_mem=44.28 GB):  45%|████▍     | 26/58 [00:05<00:01, 17.36it/s]Capturing num tokens (num_tokens=480 avail_mem=44.28 GB):  52%|█████▏    | 30/58 [00:05<00:01, 21.82it/s]Capturing num tokens (num_tokens=448 avail_mem=44.28 GB):  52%|█████▏    | 30/58 [00:05<00:01, 21.82it/s]

    Capturing num tokens (num_tokens=416 avail_mem=43.14 GB):  52%|█████▏    | 30/58 [00:05<00:01, 21.82it/s]Capturing num tokens (num_tokens=384 avail_mem=43.14 GB):  52%|█████▏    | 30/58 [00:06<00:01, 21.82it/s]

    Capturing num tokens (num_tokens=384 avail_mem=43.14 GB):  57%|█████▋    | 33/58 [00:06<00:01, 15.99it/s]Capturing num tokens (num_tokens=352 avail_mem=45.10 GB):  57%|█████▋    | 33/58 [00:06<00:01, 15.99it/s]Capturing num tokens (num_tokens=320 avail_mem=44.23 GB):  57%|█████▋    | 33/58 [00:06<00:01, 15.99it/s]Capturing num tokens (num_tokens=320 avail_mem=44.23 GB):  60%|██████    | 35/58 [00:06<00:01, 15.55it/s]Capturing num tokens (num_tokens=288 avail_mem=43.25 GB):  60%|██████    | 35/58 [00:06<00:01, 15.55it/s]

    Capturing num tokens (num_tokens=256 avail_mem=43.24 GB):  60%|██████    | 35/58 [00:06<00:01, 15.55it/s]Capturing num tokens (num_tokens=256 avail_mem=43.24 GB):  64%|██████▍   | 37/58 [00:06<00:01, 13.40it/s]Capturing num tokens (num_tokens=240 avail_mem=43.24 GB):  64%|██████▍   | 37/58 [00:06<00:01, 13.40it/s]

    Capturing num tokens (num_tokens=224 avail_mem=44.22 GB):  64%|██████▍   | 37/58 [00:06<00:01, 13.40it/s]Capturing num tokens (num_tokens=224 avail_mem=44.22 GB):  67%|██████▋   | 39/58 [00:06<00:01, 12.47it/s]Capturing num tokens (num_tokens=208 avail_mem=43.29 GB):  67%|██████▋   | 39/58 [00:06<00:01, 12.47it/s]Capturing num tokens (num_tokens=192 avail_mem=43.29 GB):  67%|██████▋   | 39/58 [00:06<00:01, 12.47it/s]

    Capturing num tokens (num_tokens=192 avail_mem=43.29 GB):  71%|███████   | 41/58 [00:06<00:01, 11.92it/s]Capturing num tokens (num_tokens=176 avail_mem=44.21 GB):  71%|███████   | 41/58 [00:06<00:01, 11.92it/s]Capturing num tokens (num_tokens=160 avail_mem=43.35 GB):  71%|███████   | 41/58 [00:06<00:01, 11.92it/s]Capturing num tokens (num_tokens=160 avail_mem=43.35 GB):  74%|███████▍  | 43/58 [00:07<00:01, 11.65it/s]Capturing num tokens (num_tokens=144 avail_mem=43.34 GB):  74%|███████▍  | 43/58 [00:07<00:01, 11.65it/s]

    Capturing num tokens (num_tokens=128 avail_mem=44.21 GB):  74%|███████▍  | 43/58 [00:07<00:01, 11.65it/s]Capturing num tokens (num_tokens=128 avail_mem=44.21 GB):  78%|███████▊  | 45/58 [00:07<00:01, 11.78it/s]Capturing num tokens (num_tokens=112 avail_mem=43.41 GB):  78%|███████▊  | 45/58 [00:07<00:01, 11.78it/s]Capturing num tokens (num_tokens=96 avail_mem=43.41 GB):  78%|███████▊  | 45/58 [00:07<00:01, 11.78it/s] 

    Capturing num tokens (num_tokens=96 avail_mem=43.41 GB):  81%|████████  | 47/58 [00:07<00:00, 11.45it/s]Capturing num tokens (num_tokens=80 avail_mem=44.19 GB):  81%|████████  | 47/58 [00:07<00:00, 11.45it/s]Capturing num tokens (num_tokens=64 avail_mem=43.66 GB):  81%|████████  | 47/58 [00:07<00:00, 11.45it/s]Capturing num tokens (num_tokens=64 avail_mem=43.66 GB):  84%|████████▍ | 49/58 [00:07<00:00, 12.01it/s]Capturing num tokens (num_tokens=48 avail_mem=43.46 GB):  84%|████████▍ | 49/58 [00:07<00:00, 12.01it/s]

    Capturing num tokens (num_tokens=32 avail_mem=44.18 GB):  84%|████████▍ | 49/58 [00:07<00:00, 12.01it/s]Capturing num tokens (num_tokens=32 avail_mem=44.18 GB):  88%|████████▊ | 51/58 [00:07<00:00, 12.28it/s]Capturing num tokens (num_tokens=28 avail_mem=43.52 GB):  88%|████████▊ | 51/58 [00:07<00:00, 12.28it/s]

    Capturing num tokens (num_tokens=24 avail_mem=44.17 GB):  88%|████████▊ | 51/58 [00:07<00:00, 12.28it/s]Capturing num tokens (num_tokens=24 avail_mem=44.17 GB):  91%|█████████▏| 53/58 [00:08<00:00,  9.29it/s]Capturing num tokens (num_tokens=20 avail_mem=43.58 GB):  91%|█████████▏| 53/58 [00:08<00:00,  9.29it/s]Capturing num tokens (num_tokens=16 avail_mem=44.17 GB):  91%|█████████▏| 53/58 [00:08<00:00,  9.29it/s]

    Capturing num tokens (num_tokens=16 avail_mem=44.17 GB):  95%|█████████▍| 55/58 [00:08<00:00, 10.24it/s]Capturing num tokens (num_tokens=12 avail_mem=43.64 GB):  95%|█████████▍| 55/58 [00:08<00:00, 10.24it/s]Capturing num tokens (num_tokens=8 avail_mem=44.16 GB):  95%|█████████▍| 55/58 [00:08<00:00, 10.24it/s] Capturing num tokens (num_tokens=8 avail_mem=44.16 GB):  98%|█████████▊| 57/58 [00:08<00:00, 11.13it/s]Capturing num tokens (num_tokens=4 avail_mem=43.66 GB):  98%|█████████▊| 57/58 [00:08<00:00, 11.13it/s]

    Capturing num tokens (num_tokens=4 avail_mem=43.66 GB): 100%|██████████| 58/58 [00:08<00:00,  6.90it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:38742


Set the default backend. Note: Besides the local server, you may use also `OpenAI` or other API endpoints.


```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-21 00:33:23] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


## Basic Usage

The most simple way of using SGLang frontend language is a simple question answer dialog between a user and an assistant.


```python
@function
def basic_qa(s, question):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user(question)
    s += assistant(gen("answer", max_tokens=512))
```


```python
state = basic_qa("List 3 countries and their capitals.")
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>Sure! Here are three countries and their respective capitals:<br><br>1. **France** - Paris<br>2. **India** - New Delhi<br>3. **Japan** - Tokyo</strong>


## Multi-turn Dialog

SGLang frontend language can also be used to define multi-turn dialogs.


```python
@function
def multi_turn_qa(s):
    s += system(f"You are a helpful assistant than can answer questions.")
    s += user("Please give me a list of 3 countries and their capitals.")
    s += assistant(gen("first_answer", max_tokens=512))
    s += user("Please give me another list of 3 countries and their capitals.")
    s += assistant(gen("second_answer", max_tokens=512))
    return s


state = multi_turn_qa()
print_highlight(state["first_answer"])
print_highlight(state["second_answer"])
```


<strong style='color: #00008B;'>Sure! Here is a list of 3 countries along with their capitals:<br><br>1. **France** - Paris<br>2. **Germany** - Berlin<br>3. **Japan** - Tokyo</strong>



<strong style='color: #00008B;'>Of course! Here is another list of 3 countries along with their capitals:<br><br>1. **Italy** - Rome<br>2. **Canada** - Ottawa<br>3. **Mexico** - Mexico City</strong>


## Control flow

You may use any Python code within the function to define more complex control flows.


```python
@function
def tool_use(s, question):
    s += assistant(
        "To answer this question: "
        + question
        + ". I need to use a "
        + gen("tool", choices=["calculator", "search engine"])
        + ". "
    )

    if s["tool"] == "calculator":
        s += assistant("The math expression is: " + gen("expression"))
    elif s["tool"] == "search engine":
        s += assistant("The key word to search is: " + gen("word"))


state = tool_use("What is 2 * 2?")
print_highlight(state["tool"])
print_highlight(state["expression"])
```


<strong style='color: #00008B;'>calculator</strong>



<strong style='color: #00008B;'>2 * 2. <br><br>Let's solve it:<br><br>2 * 2 = 4<br><br>So, 2 * 2 equals 4. You didn't need a calculator for this simple multiplication, but I can confirm the result is correct.</strong>


## Parallelism

Use `fork` to launch parallel prompts. Because `sgl.gen` is non-blocking, the for loop below issues two generation calls in parallel.


```python
@function
def tip_suggestion(s):
    s += assistant(
        "Here are two tips for staying healthy: "
        "1. Balanced Diet. 2. Regular Exercise.\n\n"
    )

    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += assistant(
            f"Now, expand tip {i+1} into a paragraph:\n"
            + gen("detailed_tip", max_tokens=256, stop="\n\n")
        )

    s += assistant("Tip 1:" + forks[0]["detailed_tip"] + "\n")
    s += assistant("Tip 2:" + forks[1]["detailed_tip"] + "\n")
    s += assistant(
        "To summarize the above two tips, I can say:\n" + gen("summary", max_tokens=512)
    )


state = tip_suggestion()
print_highlight(state["summary"])
```


<strong style='color: #00008B;'>1. **Balanced Diet:**<br>   - Consume a variety of foods from all food groups in appropriate proportions.<br>   - Include fruits, vegetables, whole grains, lean proteins, and healthy fats.<br>   - Limit processed foods, sugary drinks, excessive sodium, and unhealthy fats.<br>   - Eat regularly to maintain energy levels and manage weight.<br><br>2. **Regular Exercise:**<br>   - Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week.<br>   - Include strength training exercises at least two days a week.<br>   - Find activities you enjoy to maintain consistency and motivation.<br><br>These tips help ensure that you maintain overall health and wellness.</strong>


## Constrained Decoding

Use `regex` to specify a regular expression as a decoding constraint. This is only supported for local models.


```python
@function
def regular_expression_gen(s):
    s += user("What is the IP address of the Google DNS servers?")
    s += assistant(
        gen(
            "answer",
            temperature=0,
            regex=r"((25[0-5]|2[0-4]\d|[01]?\d\d?).){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
        )
    )


state = regular_expression_gen()
print_highlight(state["answer"])
```


<strong style='color: #00008B;'>208.67.222.222</strong>


Use `regex` to define a `JSON` decoding schema.


```python
character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n"""
    + r"""        "wood": "[\w\d\s]{1,16}",\n"""
    + r"""        "core": "[\w\d\s]{1,16}",\n"""
    + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "alive": "(Alive|Deceased)",\n"""
    + r"""    "patronus": "[\w\d\s]{1,16}",\n"""
    + r"""    "bogart": "[\w\d\s]{1,16}"\n"""
    + r"""\}"""
)


@function
def character_gen(s, name):
    s += user(
        f"{name} is a character in Harry Potter. Please fill in the following information about this character."
    )
    s += assistant(gen("json_output", max_tokens=256, regex=character_regex))


state = character_gen("Harry Potter")
print_highlight(state["json_output"])
```


<strong style='color: #00008B;'>{<br>    "name": "Harry Potter",<br>    "house": "Gryffindor",<br>    "blood status": "Half-blood",<br>    "occupation": "student",<br>    "wand": {<br>        "wood": " Cherry",<br>        "core": "Phoenix feather",<br>        "length": 11.0<br>    },<br>    "alive": "Alive",<br>    "patronus": "Stag",<br>    "bogart": "Old Lady"<br>}</strong>


## Batching 

Use `run_batch` to run a batch of prompts.


```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


states = text_qa.run_batch(
    [
        {"question": "What is the capital of the United Kingdom?"},
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Japan?"},
    ],
    progress_bar=True,
)

for i, state in enumerate(states):
    print_highlight(f"Answer {i+1}: {states[i]['answer']}")
```

      0%|          | 0/3 [00:00<?, ?it/s]

     33%|███▎      | 1/3 [00:00<00:00,  7.45it/s]

     67%|██████▋   | 2/3 [00:00<00:00, 11.81it/s]

    100%|██████████| 3/3 [00:00<00:00, 18.62it/s]

    



<strong style='color: #00008B;'>Answer 1: The capital of the United Kingdom is London.</strong>



<strong style='color: #00008B;'>Answer 2: The capital of France is Paris.</strong>



<strong style='color: #00008B;'>Answer 3: The capital of Japan is Tokyo.</strong>


## Streaming 

Use `stream` to stream the output to the user.


```python
@function
def text_qa(s, question):
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


state = text_qa.run(
    question="What is the capital of France?", temperature=0.1, stream=True
)

for out in state.text_iter():
    print(out, end="", flush=True)
```

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    What is the capital of France?<|im_end|>
    <|im_start|>assistant


    The

     capital

     of

     France

     is

     Paris

    .

    <|im_end|>


## Complex Prompts

You may use `{system|user|assistant}_{begin|end}` to define complex prompts.


```python
@function
def chat_example(s):
    s += system("You are a helpful assistant.")
    # Same as: s += s.system("You are a helpful assistant.")

    with s.user():
        s += "Question: What is the capital of France?"

    s += assistant_begin()
    s += "Answer: " + gen("answer", max_tokens=100, stop="\n")
    s += assistant_end()


state = chat_example()
print_highlight(state["answer"])
```


<strong style='color: #00008B;'> The capital of France is Paris.</strong>



```python
terminate_process(server_process)
```

## Multi-modal Generation

You may use SGLang frontend language to define multi-modal prompts.
See [here](https://docs.sglang.io/supported_models/text_generation/multimodal_language_models.html) for supported models.


```python
server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --log-level warning"
)

wait_for_server(f"http://localhost:{port}", process=server_process)
print(f"Server started on http://localhost:{port}")
```

    /actions-runner/_work/sglang/sglang/python/sglang/launch_server.py:54: UserWarning: 'python -m sglang.launch_server' is still supported, but 'sglang serve' is the recommended entrypoint.
      Example: sglang serve --model-path <model> [options]
      warnings.warn(


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-21 00:33:43] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
    [2026-05-21 00:33:47] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.


    Multi-thread loading shards:   0% Completed | 0/5 [00:00<?, ?it/s]

    Multi-thread loading shards:  20% Completed | 1/5 [00:00<00:03,  1.15it/s]

    Multi-thread loading shards:  40% Completed | 2/5 [00:01<00:02,  1.16it/s]

    Multi-thread loading shards:  60% Completed | 3/5 [00:02<00:01,  1.18it/s]

    Multi-thread loading shards:  80% Completed | 4/5 [00:03<00:00,  1.20it/s]

    Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.54it/s]Multi-thread loading shards: 100% Completed | 5/5 [00:03<00:00,  1.35it/s]


    /usr/local/lib/python3.10/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
      response = await f(request)


    2026-05-21 00:34:02,019 - CUTE_DSL - WARNING - [handle_import_error] - Unexpected error during package walk: cutlass.cute.experimental
    [2026-05-21 00:34:02] Unexpected error during package walk: cutlass.cute.experimental



<strong style='color: #00008B;'><br><br>        NOTE: Typically, the server runs in a separate terminal.<br>        In this notebook, we run the server and notebook code together, so their outputs are combined.<br>        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.<br>        To reduce the log length, we set the log level to warning for the server, the default log level is info.<br>        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.<br>        </strong>


    Server started on http://localhost:30197



```python
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))
```

    [2026-05-21 00:34:04] Endpoint '/get_model_info' is deprecated and will be removed in a future version. Please use '/model_info' instead.


Ask a question about an image.


```python
@function
def image_qa(s, image_file, question):
    s += user(image(image_file) + question)
    s += assistant(gen("answer", max_tokens=256))


image_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
image_bytes, _ = load_image(image_url)
state = image_qa(image_bytes, "What is in the image?")
print_highlight(state["answer"])
```

    /actions-runner/_work/sglang/sglang/python/sglang/srt/utils/common.py:877: UserWarning: The given buffer is not writable, and PyTorch does not support non-writable tensors. This means you can write to the underlying (supposedly non-writable) buffer using the tensor. You may want to copy the buffer to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:1586.)
      encoded_image = torch.frombuffer(image_bytes, dtype=torch.uint8)



<strong style='color: #00008B;'>The image shows a person performing an unusual task in an urban setting. The individual appears to be ironing clothes while standing on the back of a moving taxi. The scene seems unexpected and possibly spontaneous, as taxis are typically used for transportation and not for parking. The person is balancing as they iron, indicating a precarious and unconventional activity taking place in a public space. The background consists of city buildings and a yellow taxi, which is common in many urban areas, suggesting this action may be happening in a major city. The person is also wearing a yellow shirt and was previously engaged in another activity, possibly juggling, as also seen in the perspective of the image.</strong>



```python
terminate_process(server_process)
```
