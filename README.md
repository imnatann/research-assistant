# Asisten Riset Bilingual

Proyek CLI ini mencari literatur, memeriksa dukungan sumber untuk setiap klaim, dan menghasilkan draf paper dalam bahasa Inggris, Indonesia, atau keduanya. Inferensi lokal memakai llama.cpp dan Metal pada Mac Apple Silicon.

## Pemasangan

Jalankan perintah berikut dari root proyek di Mac Apple Silicon. `./research` memerlukan Python 3.9 atau lebih baru. `./wsp setup` memerlukan `uv` dan menyiapkan Python 3.12. Evaluasi coding dan benchmark agentic juga memerlukan Docker Desktop.

```sh
mkdir -p .runtime models
curl -fL --retry 3 -o .runtime/llama-b12359-bin-macos-arm64.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b12359/llama-b12359-bin-macos-arm64.tar.gz
printf '%s  %s\n' 82b00abc740160e1b9980e6cc79c117345fa34af058c80e6c0a69aa8ddccde75 \
  .runtime/llama-b12359-bin-macos-arm64.tar.gz | shasum -a 256 -c - &&
  tar -xzf .runtime/llama-b12359-bin-macos-arm64.tar.gz -C .runtime

curl -fL --retry 3 -o models/LFM3-1.7B-Q4_K_M.gguf.part \
  https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/cd0c74c575b5c416f3011b5b4c298077f26ca8c2/LFM3-1.7B-Q4_K_M.gguf
printf '%s  %s\n' 3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4 \
  models/LFM3-1.7B-Q4_K_M.gguf.part | shasum -a 256 -c - &&
  mv -n models/LFM3-1.7B-Q4_K_M.gguf.part models/LFM3-1.7B-Q4_K_M.gguf

./research doctor
```

[Model dan bobot resmi](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) · [Rilis runtime](https://github.com/ggml-org/llama.cpp/releases/tag/b12359) · [Lisensi model](models/LICENSE-Liquid-AI.txt)

## Weight-space programming

`./wsp` menyediakan lingkungan Python 3.12 terpisah untuk membaca dan mengedit bobot LFM3-1.7B langsung dari GGUF, lalu membandingkan hasilnya dengan probe bilingual proyek. Lihat [catatan riset dan panduan eksperimen](docs/weight-space.md).

```sh
./wsp setup
./wsp list --prefix blk.23
./wsp inspect blk.23.ffn_gate.weight --row 0 --count 8
eksperimen="$(./wsp patch examples/wsp-ablation.json)"
./wsp compare "$eksperimen"
```

Untuk agentic coding, pakai `./wsp bench setup` lalu `./wsp bench both --smoke`. Runner memakai tugas resmi terpilih dari Terminal-Bench 4.1.0 dan DeepSWE 1.1. Lihat [panduan benchmark agentic coding](docs/agentic-benchmarks.md).

K2-Horizon-7B Q4_K_M dapat dipasang untuk tugas coding. Pada set uji lokal 24 soal Python, K2 lulus 15 soal dan LFM3 lulus 10. [Hasil dan perintah reproduksi](docs/coding-upgrade.md)

## Menjalankan riset

```sh
./research doctor
./research run "How does ALCE measure whether citations support generated answers?" \
  --arxiv 2305.14627 --lang both
```

Pencarian default mengirim kata kunci ke arXiv dan OpenAlex. Gunakan `--query` untuk memilih frasa sendiri, `--arxiv ID` berulang untuk menetapkan sumber, atau `--sources` untuk memakai catatan lokal secara offline. `--lang en` dan `--lang id` memilih satu bahasa. Atur `--limit` dari 1 sampai 6.

```sh
./research run "Seberapa baik model bahasa mendukung klaim dengan sitasi?" \
  --query "language model citation generation" --mode both --limit 2

./research run "Did the audit improve citation support in this example?" \
  --sources examples/sources.json --lang both
```

Contoh offline memakai catatan sintetis dengan angka ilustratif untuk menguji alur kerja. Untuk catatan sendiri, gunakan struktur yang sama dan isi URL HTTPS, judul, bagian, dan cuplikan yang benar. Simpan DOI atau arXiv ID jika tersedia.

## Hasil

Setiap perintah `run` membuat direktori tersendiri di `runs/`. File yang dihasilkan meliputi

- `manuscript.en.md` dan `manuscript.id.md` dengan tautan sitasi yang dirakit dari sumber yang ditemukan.
- `evidence.json` dengan klaim, cuplikan yang dipakai, lokasi, dan penilaian dukungan.
- `review.*.json` dengan penilaian per paragraf dan alasan ketika buktinya lemah. Paragraf itu juga ditandai di draf.
- `sources.json` dan `references.bib` dengan catatan sumber dan metadata.
- `hypotheses.*.json` dengan hipotesis, studi terdekat dari sumber yang tersedia, perbedaan yang diusulkan, uji, dan kondisi yang dapat membantahnya.
- `run.json` dan `trace.jsonl` dengan konfigurasi, skor pemeriksaan otomatis, dan riwayat tahapan model.

OpenAlex memberi abstrak ketika tersedia. Untuk arXiv, harness membaca abstrak dan mengambil beberapa cuplikan tambahan bila HTML full text tersedia. Laporan tidak mengklaim pencarian literatur menyeluruh atau menyatakan hipotesis baru telah terbukti. Pemeriksaan paragraf memakai model lokal yang sama dalam konteks terpisah. Tinjau sumber asli, angka, dan kelengkapan literatur sebelum memakai draf untuk publikasi.

Mode `both` menggunakan pertanyaan, sumber, dan pengaturan sampling yang sama. Draf `baseline/` memakai semua klaim yang diekstrak. Draf `audited/` memakai klaim yang lolos pemeriksaan sebelum penulisan. Jika sebagian klaim didukung, model dapat mempersempit klaim itu dan mengaudit revisinya sebelum dipakai. Dua draf mendapat pemeriksaan paragraf yang sama. Gunakan peninjauan buta untuk mengukur perubahan akurasi sitasi, cakupan temuan, dan kualitas bahasa.

`./research probe` menjalankan empat skenario sintetis untuk mengecek apakah model mengikuti bukti saat didesak atau diberi informasi baru. Laporan menyimpan jawaban lengkap untuk pemeriksaan manual. Empat skenario ini cukup untuk smoke test, sehingga penilaian yang lebih luas memerlukan topik dan formulasi tambahan.

## Iterasi dan data LoRA

```sh
./research improve runs/NAMA-RUN/run.json \
  --feedback "Draf Indonesia terlalu yakin pada hasil yang hanya didukung abstrak."
```

Perintah ini menyimpan satu usulan perubahan dan rancangan uji di `improvements/` di dalam run. Terapkan usulan setelah menilainya pada set tugas terpisah yang model belum lihat. Launcher tidak mengubah bobot model atau strategi yang sedang dipakai.

Untuk training, mulai dari checkpoint post-trained [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) dan [SFT LoRA lewat TRL](https://docs.liquid.ai/lfm/fine-tuning/trl). Kurasi `trace.jsonl` dengan revisi manusia sebelum dijadikan data. Siapkan tulisan asli berbahasa Inggris dan Indonesia. Buat pasangan uji yang mengukur apakah model mempertahankan kesimpulan saat ditekan tanpa bukti dan mengubahnya saat ada bukti baru yang relevan. Pisahkan evaluasi berdasarkan topik dan sumber agar paper yang sama tidak muncul di data latihan dan uji.
