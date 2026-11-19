# Evaluasi runtime dan jalur training

Catatan ini memuat hasil pengukuran runtime pada Windows 11, AMD Ryzen 9 9900X, dan NVIDIA RTX 4060 Ti 16 GB. Semua angka berasal dari `llama-bench` dengan tiga pengulangan pada `models/LFM3-1.7B-Q4_K_M.gguf`, seluruh lapisan di GPU (`-ngl 99`).

## Hasil pengukuran

Baseline `llama-bench` b12359: prefill 14.445 t/s (pp512), decode 196 t/s (tg128).

| Konfigurasi | pp512 | Δ prefill | tg128 | Δ decode |
| --- | --- | --- | --- | --- |
| default | 14.445 | — | 196,2 | — |
| `-fa on` | 15.153 | **+4,9%** | 195,3 | −0,5% |
| `-fa on -ub 1024` | 14.932 | +3,4% | 195,9 | −0,2% |
| `-fa on -ub 2048` | 14.941 | +3,4% | 195,9 | −0,2% |
| `-fa on -ctk/-ctv q8_0` | 14.666 | +1,5% | 193,8 | −1,2% |
| `-fa on -lm mlock` | 14.923 | +3,3% | 196,2 | ±0 |
| `-fa on -lzm off` | 14.890 | +3,1% | 195,9 | −0,2% |

Build v1.0.0 terbaru (b12532) diukur dengan cara yang sama: default 14.376 t/s dan `-fa on` 15.085 t/s, yaitu rasio yang sama dengan b12359. Build lebih baru tidak memberi kecepatan tambahan, sehingga pemasangan tetap memakai b12359 seperti yang dirujuk dokumentasi.

Backend Vulkan juga diuji sebagai alternatif CUDA. Pada campuran realistis (`-pg 1024,512`), CUDA 574,7 t/s dan Vulkan 572,4 t/s, sehingga selisihnya berada dalam rentang variasi antar-pengulangan. Selisih decode yang terlihat pada uji terpisah (205 vs 199 t/s) tidak muncul lagi saat prefill dan decode diukur bersama.

Kesimpulan: untuk checkpoint ini, perubahan build atau backend tidak mempercepat inferensi secara berarti. Satu-satunya perbaikan yang terukur adalah mengaktifkan flash attention secara eksplisit, dan `research_assistant/model.py` sekarang mengirim `-fa on` ke `llama-server`. Flag itu menaikkan prefill sekitar 5% dan tidak mengubah decode, dengan `research doctor` tetap lulus.

Perlu dicatat bahwa kenaikan prefill 5% itu **tidak terlihat pada run ujung-ke-ujung**: run ALCE yang sama memakai 24,93 detik waktu model, dibandingkan 24,96 detik sebelum flag dipasang. Alasannya ada di bagian berikut.

## Ke mana waktu run sebenarnya pergi

Run ALCE dua bahasa memakai 24,96 detik waktu model untuk 19 panggilan berurutan. Rinciannya: `extract_claims` 2,55 s, `review_evidence` 3,66 s, `draft_en` 1,61 s, `review_draft_en` 1,58 s, `draft_id` 2,14 s, `review_draft_id` 2,59 s, `hypotheses` 3,48 s, dan sisanya delapan panggilan lokalisasi berdurasi 0,3–1,0 s. Server menyala dalam 1,06 detik, jadi startup bukan penyebab.

Artinya beban ada pada puluhan panggilan decode pendek yang saling bergantung, bukan pada prefill. Mempercepat prefill 5% hanya menghemat sebagian kecil dari total. Yang lebih berpengaruh:

- Panggilan lokalisasi Indonesia berjalan satu per satu meski tidak saling bergantung. Menjalankannya bersamaan akan memangkas bagian itu.
- `--limit` dan jumlah paragraf menentukan panjang prompt; menurunkannya langsung memotong waktu decode.
- Server sudah `-np 1`; menaikkan slot paralel hanya membantu bila panggilan benar-benar dikirim bersamaan.

## Jalur training di Windows

Resep training saat ini memakai MLX dan MLX-LM, sehingga terikat pada macOS. Di Windows jalur itu tidak tersedia, dan inti paketnya (`uv pip install 'mlx==0.32.4'`) juga hanya menerbitkan roda untuk Apple Silicon.

Jalur penggantinya sudah diuji dan berfungsi end-to-end memakai Unsloth. Unsloth 2026.11.6 terpasang native tanpa WSL (torch 2.14.1+cu132, CUDA 8.9, Triton 3.9.0), dan **LFM3 memang dikenali**: Unsloth mencetak `Fast Lfm3 patching` lalu memperbaiki kode remote-nya (`Repaired remote modeling code so it trains: Lfm3ForCausalLM.forward`). Katalog model Unsloth tidak memuat LFM3, tetapi jalur patch-nya jelas ada.

Hasil pengujian pada perangkat ini:

| Tahap | Hasil |
| --- | --- |
| Muat 4-bit | `Lfm3ForCausalLM`, 1,21 GB VRAM |
| Pasang LoRA (r8/alpha16, `w1`/`w2`/`w3`, 12 layer terakhir) | 2.801.664 parameter |
| Forward pass | logits `(1, 22, 128000)` |
| Training SFT 8 langkah | loss akhir 1,2301 |
| Puncak VRAM | 1,30–1,35 GB |
| Ekspor PEFT ke GGUF | 72 tensor, 5,6 MB |
| Muat di llama.cpp | `loaded 72 tensors from lora file` |

Angka 2.801.664 parameter itu penting: dokumentasi MLX menyebut 2,802 juta parameter LoRA pada MLP 12 layer terakhir, jadi resep Windows ini mereproduksi konfigurasi macOS secara identik dan hasilnya bisa dibandingkan langsung.

### Kendala versi yang harus diketahui

Ini bagian yang paling mudah menjebak, dan urutan pemasangan menentukan:

1. **Pemasangan.** Di Windows, `torchao` harus dipasang lebih dulu. Tanpa itu `uv pip install unsloth --torch-backend=auto` diam-diam memasang Unsloth versi lama. Lingkungan dipisah di `.runtime/unsloth/venv` agar tidak mengganggu dependensi proyek.
2. **Versi `transformers`.** LFM3 membutuhkan dua hal yang bertabrakan pada seri 5.12–5.17:
   - kode model mengimpor `use_kernel_func_from_hub_with_fallback`, yang hanya ada di `transformers` 5.15 ke atas;
   - di 5.15 ke atas, `layer_types` dinormalkan dari `"conv"` menjadi `"linear_attention"`, sedangkan `create_masks_by_layer_type` membuat kunci peta mask berdasarkan `"conv"`, sehingga forward gagal dengan `KeyError: 'linear_attention'`.

   Kombinasi yang bekerja: `transformers==5.14.0` ditambah shim dekorator identitas untuk simbol yang hilang. Shim itu aman karena simbol tersebut hanya membungkus kernel GPU `causal_conv1d` opsional; jalur murni PyTorch dipakai baik ada maupun tidak ada kernel.

### Menjalankan training

```powershell
python -m research_assistant.train_unsloth data\train.jsonl `
  --out runs\unsloth-lora\adapter --steps 64
```

Data berformat JSONL dengan `{"text": ...}` atau `{"prompt": ..., "completion": ...}`. Berkas dengan BOM UTF-8 dari PowerShell tetap terbaca. Versi yang dipin ada di `requirements-unsloth.txt`; jangan menaikkan `transformers` tanpa mengulang uji.

### Ekspor ke llama.cpp

`lora_export.py` tidak diperlukan di jalur ini. Konverter resmi llama.cpp menerima adapter PEFT langsung:

```powershell
python convert_lora_to_gguf.py <base-model-dir> --outtype f16 `
  --outfile adapter.gguf <adapter-dir>
```

Nama tensor tetap gaya LFM (`feed_forward.w1` menjadi `ffn_gate`), bukan gaya Qwen `mlp.gate_proj`, sehingga tidak ada pemetaan nama yang perlu ditulis sendiri.

### Catatan

- Peringatan `xformers is installed but cannot load with torch 2.14.1+cu132` bersifat kosmetik; PyTorch attention yang dipakai.
- `-fa on` pada llama.cpp tidak berkaitan dengan FA2 di Unsloth; keduanya jalur terpisah.
- Adapter uji di atas dilatih dari 4 contoh selama beberapa langkah, jadi hanya membuktikan mekanismenya berjalan, bukan mutu model. Penilaian mutu perlu dataset MBPP dan protokol uji yang sama seperti jalur MLX.
- Klaim kecepatan Unsloth (2× lebih cepat, 70% lebih sedikit VRAM) berasal dari tolok ukur mereka pada GPU datacenter untuk arsitektur yang sudah dipatch penuh. Untuk LFM3 belum ada angka resmi, jadi jangan mengandalkannya. Yang terukur di sini hanya penggunaan VRAM.
