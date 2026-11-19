# Weight-space programming untuk LFM3-1.7B

## Temuan riset

Weight-space programming (WSP) meminta agen membaca, menulis, atau memperbaiki bobot jaringan secara langsung. Agen bisa memakai NumPy untuk menghitung perubahan tensor dan melakukan forward pass untuk menguji hipotesis. Adaptasi bobot terjadi lewat edit eksplisit tanpa gradient descent. Takehi dkk. melaporkan bahwa edit pada dua MLP Qwen3.5-0.8B mengurangi loop dan menaikkan MATH-500 dari 7,9% ke 48,1%. Percobaan mereka pada Gemma 3 270M, yang jarang mengalami loop, menghasilkan perubahan sekitar satu poin atau kurang. [Takehi dkk., November 2026](https://rikiyat.github.io/blog/weight-space-programming/)

Hu dkk. menemukan kenaikan yang terlokalisasi dan transfer yang bervariasi antarbenchmark. Enam dari 16 percobaan mereka mempertahankan bobot awal karena kandidat perbaikan belum lolos validasi. WeightBench juga melaporkan bahwa agen frontier lebih berhasil membaca jaringan kecil dan menulis kebijakan kontrol daripada memperbaiki model bahasa pada tugas benchmark tersebut. [Hu dkk.](https://arxiv.org/abs/2611.09382) · [WeightBench](https://arxiv.org/abs/2611.18553)

Interpretasi dari bobot perlu diuji terhadap perilaku. Ahmed menemukan bahwa pengacakan urutan label menggagalkan interpretasi kelas walaupun agen masih melihat struktur pada output. Replikasi Alan Turing Institute juga menemukan bahwa pembacaan bobot jaringan kecil berhasil terutama untuk struktur yang tampak dekat input. Untuk LFM3, dugaan fungsi neuron sebaiknya diuji dengan intervensi yang diprediksi sebelumnya dan kontrol yang sebanding. [Ahmed](https://arxiv.org/abs/2611.08070) · [Alan Turing Institute](https://www.turing.ac.uk/news/publications/replicating-weight-space-programming-writing-and-reading-neural-network-weights)

## Lingkungan proyek

Target eksperimen adalah LFM3-1.7B post-trained. Ikuti [petunjuk pemasangan](../README.md#pemasangan) untuk menyimpan checkpoint di `models/LFM3-1.7B-Q4_K_M.gguf`. Model memiliki 24 blok yang menggabungkan short convolution, sliding-window attention, dan global attention. Inventaris checkpoint lokal berisi 287 tensor (136 Q4_K, 18 Q6_K, dan 133 F32). Susunan blok dan fungsi tensor perlu dilacak pada arsitektur LFM3 sebelum memindahkan hipotesis edit dari Qwen. [Model card Liquid AI](https://huggingface.co/LiquidAI/LFM3-1.7B) · [GGUF resmi](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF)

Jalankan perintah dari root proyek. `./wsp setup` memakai `uv` untuk memasang dependensi yang dipin di `.venv-wsp/`. Inferensi tetap memakai llama.cpp b12359 dan Metal. Setelah pemasangan, alur ini berjalan lokal memakai checkpoint yang sudah tersedia.

```sh
./wsp setup
./wsp list --prefix blk.23
./wsp inspect blk.23.ffn_gate.weight --row 0 --start 0 --count 8
eksperimen="$(./wsp patch examples/wsp-ablation.json)"
./wsp compare "$eksperimen"
./wsp inspect blk.23.ffn_gate.weight --model "$eksperimen/edited.gguf" --row 0 --count 8
```

Rencana JSON menyimpan SHA-256 checkpoint sumber, hipotesis, dan daftar edit. Untuk setiap edit, `ranges` memakai indeks mulai dari nol dengan batas akhir eksklusif dalam urutan NumPy. Matriks memakai `[[baris_awal, baris_akhir], [kolom_awal, kolom_akhir]]` dan vektor memakai satu rentang. Operasi `set`, `add`, dan `scale` menerima skalar atau array numerik dengan bentuk yang sama dengan irisan. Edit diterapkan sesuai urutan daftar. Agen atau peneliti menentukan nilai edit setelah meninjau bobot dan perilaku.

`patch` memverifikasi SHA-256 resmi dan menulis `edited.gguf` serta `patch.json` ke direktori eksperimen baru di `runs/weight-space/`. Untuk target Q4_K atau Q6_K, tensor yang diedit didekuantisasi ke F32 dan diekspor dalam tipe itu, sementara payload semua tensor lain disalin. Dekuantisasi hanya merekonstruksi nilai dari checkpoint terkuantisasi. `control.gguf` menyimpan konversi F32 yang sama dengan nilai sebelum edit sehingga perbandingan kontrol dan hasil edit memakai tipe tensor yang sama. Untuk contoh ini siapkan sekitar 2,2 GB ekstra untuk kedua checkpoint. [Pustaka GGUF llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py)

`compare` memeriksa hash checkpoint lalu menjalankan empat kasus probe yang sama pada baseline, kontrol (jika ada), dan hasil edit. Direktori `evaluations/` menyimpan skor per kasus, jawaban, jejak prompt, dan log runtime. Skor saat ini memeriksa dua label boolean yang dihasilkan model. Tinjau teks jawaban untuk memeriksa konsistensi penjelasan dan bahasa.

## Eksperimen pertama

Contoh `examples/wsp-ablation.json` mengatur 128 baris awal pada matriks gate FFN blok terakhir menjadi nol untuk menguji penulisan bobot dan inferensi. Uji lokal mengubah 262.139 nilai dan memverifikasi bahwa metadata serta 286 tensor lain identik. Baseline, kontrol F32, dan hasil edit masing-masing lulus 3 dari 4 kasus. Ketiganya gagal pada kasus desakan berbahasa Inggris.

Untuk eksperimen kualitas asisten riset, mulai dengan satu kegagalan yang bisa direproduksi, misalnya ketika model mengubah kesimpulan setelah didesak tanpa bukti. Ukur frekuensinya pada contoh pengembangan dan karakterisasi perubahan aktivasi sebelum memilih irisan bobot. Uji patch pada topik dan sumber yang ditahan, dengan pasangan kasus desakan dan pembaruan bukti dalam kedua bahasa. Nilai ketepatan klaim, validitas sitasi, dan kepatuhan bahasa bersama skor probe.
