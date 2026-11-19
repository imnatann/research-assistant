# Evaluasi model coding lokal

K2-Horizon-7B Q4_K_M lulus **15/24** soal Python pada set uji yang sama dengan LFM3-1.7B Q4_K_M, yang lulus **10/24**. K2 lulus delapan soal yang gagal pada LFM3, dan gagal tiga soal yang lulus pada LFM3. Selisih bersihnya 20,8 poin persentase pada subset ini.

| Checkpoint | Tulis fungsi Python | Perbaiki fungsi dengan satu panggilan `edit` |
| --- | --- | --- |
| LFM3-1.7B Q4_K_M | 10/24 | 5/12 |
| LFM3-1.7B Q4_K_M dengan awalan kode Python | 12/24 | — |
| LFM3-1.7B Q4_K_M + QLoRA, langkah 64 | 7/24 | 4/12 |
| LFM3-1.7B Q4_K_M + QLoRA, langkah 256 | 7/24 | 4/12 |
| LFM3-1.7B QAD-Q4_0 | 13/24 | 4/12 |
| K2-Horizon-7B Q4_K_M | 15/24 | 7/12 |

Daftar ID soal, jumlah yang lulus, hash checkpoint, dan konfigurasi trial tersedia sebagai [data hasil benchmark](benchmark-results.json).

Evaluator memilih 24 soal dari partisi uji MBPP Sanitized dengan seed 917. Setiap prompt menyertakan satu contoh dan verifier menjalankan seluruh assert dengan Docker tanpa jaringan. Verifier meluluskan 24/24 solusi referensi. Uji edit memakai 12 soal dari subset yang sama dan menilai satu perbaikan pada fungsi yang sengaja dikosongkan. [Paper MBPP](https://arxiv.org/abs/2108.07732) · [Snapshot dataset](https://github.com/google-research/google-research/tree/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp)

## Menulis Python dengan LFM3

`./wsp code complete` membuka respons LFM3 dengan awalan blok Python dan mencetak kode ke stdout setelah pemeriksaan sintaks. Perintah ini memakai checkpoint Q4_K_M bawaan. Untuk soal pada subset lokal, jumlah respons dengan kesalahan sintaks turun dari 11 menjadi 1 dan jumlah solusi yang lulus naik dari 10 menjadi 12. Evaluasi ini memakai tugas Python pendek.

```sh
./wsp setup
./wsp code setup
./wsp code complete 'Tulis fungsi Python sum_even(numbers) yang menjumlahkan semua bilangan genap.'
./wsp code eval --prefill-python
```

Untuk menguji QAD, unduh varian resmi LFM3 QAD-Q4_0 dan simpan di `models/LFM3-1.7B-QAD-Q4_0.gguf`. Ukurannya 1.049.615.680 byte dan SHA-256-nya `43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452`. Hasil 13/24 untuk QAD memakai protokol chat yang sama dengan baseline 10/24. [Checkpoint Liquid AI](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/blob/cd0c74c575b5c416f3011b5b4c298077f26ca8c2/LFM3-1.7B-QAD-Q4_0.gguf)

## Menjalankan K2-Horizon

Unduh checkpoint resmi ke `models/K2-Horizon-7B-Q4_K_M.gguf`. Ukurannya 5.592.217.984 byte dan SHA-256-nya `eb89c15a0ae9712be2ee462bf43802de14200f20f93b73da6eb68c2ebdd28e4e`. Uji OpenCode memakai cache KV Q4_0 agar checkpoint dan konteks muat pada Mac 16 GB. [Checkpoint IFM](https://huggingface.co/IFM/K2-Horizon-7B-GGUF/blob/7a75dbda05e3d440351543131ff3f7fa5687e879/K2-Horizon-7B-Q4_K_M.gguf)

```sh
./wsp setup
./wsp code setup
./wsp code eval --gguf models/K2-Horizon-7B-Q4_K_M.gguf
./wsp bench setup
caffeinate -i ./wsp bench terminal-bench --smoke --model models/K2-Horizon-7B-Q4_K_M.gguf \
  --context-tokens 32768 --kv-cache q4_0 --agent-seconds 420 --reasoning-budget 512 \
  --edit-test-workflow
caffeinate -i ./wsp bench deepswe --smoke --model models/K2-Horizon-7B-Q4_K_M.gguf \
  --context-tokens 65536 --kv-cache q4_0 --agent-seconds 420 --reasoning-budget 512
```

Untuk menggunakan K2 sebagai endpoint lokal di luar benchmark, jalankan perintah ini dari root proyek. Endpoint `/v1/chat/completions` tersedia pada `127.0.0.1:8080`.

```sh
.runtime/llama-b12359/llama-server -m models/K2-Horizon-7B-Q4_K_M.gguf \
  --alias K2-Horizon-7B-Q4_K_M --host 127.0.0.1 --port 8080 \
  -c 32768 -np 1 -ngl 99 --jinja --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 --no-webui
```

## Hasil benchmark agentic

| Suite | LFM3 bawaan | K2-Horizon |
| --- | --- | --- |
| Terminal-Bench 4.1.0 | 0/1 | 0/1 |
| DeepSWE 1.1 | 0/1 | 0/1 |

OpenCode memakai budget 420 detik per tugas. Pada DeepSWE, K2 dengan konteks 64K lulus 16/16 tes regresi dan 0/3 tes fitur. Pada uji ulang Terminal-Bench dengan konteks 32K dan instruksi edit–test, agen membuat 11 panggilan Bash dan baca serta mencapai timeout. Run awal Terminal-Bench terganggu sleep, dan satu percobaan ulang dihentikan saat dua proses inferensi tumpang tindih dan GPU kehabisan memori. Skor Terminal-Bench di tabel berasal dari ulangan dengan satu proses model. Tiap skor mencakup satu tugas resmi.

## Adaptasi LFM3

Pelatihan lokal memodifikasi 2,802 juta parameter LoRA pada MLP di 12 layer terakhir. Dataset berisi 294 contoh dari 147 soal partisi latih MBPP. Contohnya berupa fungsi Python dan panggilan `edit`. Set validasi berisi 32 contoh dari 16 soal terpisah dalam partisi latih. Model diproses dengan MLX 0.32.4 dan MLX-LM pada commit `a1de292b740bd8297e173f2c73047f5b6bdc43f1`. [Bobot MLX resmi Liquid AI](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit/tree/91f34088d6946482cbcae11b85d4657093618c34) · [Dukungan LFM3 di MLX-LM](https://github.com/ml-explore/mlx-lm/pull/2016)

Checkpoint langkah 256 dipilih berdasarkan loss validasi 0,367. Setelah diekspor ke GGUF dan dipasang pada LFM3 lokal, hasil kedua uji fungsional turun. Checkpoint langkah 64 mencapai skor fungsional yang sama dengan langkah 256. [Konfigurasi pelatihan](../examples/lfm3-coding-qlora.yaml)

```sh
uv venv --python 3.12 .runtime/coding-mlx/.venv
uv pip install --python .runtime/coding-mlx/.venv/bin/python 'mlx==0.32.4' \
  'mlx-lm @ git+https://github.com/techwritescode/mlx-lm.git@a1de292b740bd8297e173f2c73047f5b6bdc43f1'
.runtime/coding-mlx/.venv/bin/hf download LiquidAI/LFM3-1.7B-MLX-4bit \
  --revision 91f34088d6946482cbcae11b85d4657093618c34 --local-dir .runtime/coding-mlx/model
./wsp code prepare .runtime/coding-mlx/data
.runtime/coding-mlx/.venv/bin/mlx_lm.lora -c examples/lfm3-coding-qlora.yaml \
  --adapter-path runs/coding-mlx-replay/adapter
./wsp export-lora runs/coding-mlx-replay/adapter runs/coding-mlx-replay/adapter.gguf \
  --weights runs/coding-mlx-replay/adapter/0000256_adapters.safetensors
./wsp code eval --lora runs/coding-mlx-replay/adapter.gguf
./wsp code eval --repair --lora runs/coding-mlx-replay/adapter.gguf
```
