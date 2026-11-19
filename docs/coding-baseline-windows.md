# Baseline coding checkpoints (Windows)

Dokumen ini mencatat cara mengukur state model di mesin ini dan menyimpan skornya sebagai checkpoint pembanding. Tujuannya agar perubahan setelah fine-tuning bisa dinilai terhadap angka yang tetap, bukan ingatan.

## Ringkasan hasil

Checkpoint `lfm3-base-20261119` adalah state saat ini: `models/LFM3-1.7B-Q4_K_M.gguf` tanpa adapter LoRA.

| Slice | Skor | Task ID yang lulus |
| --- | --- | --- |
| write-python (24 task) | **11/24** | 79, 141, 223, 226, 230, 281, 287, 393, 406, 412, 452 |
| repair-one-tool-call (12 task) | **4/12** | 71, 77, 172, 230 |

Manifest dan skor tersimpan di `runs/checkpoints/`:

- `lfm3-base-20261119.json` memuat SHA-256 checkpoint, ukuran, runtime, argumen server, dan host.
- `lfm3-base-20261119-score.json` memuat skor per slice dan daftar task yang lulus.

`runs/` diabaikan git, jadi berkas ini ada di disk dan tidak ikut commit.

## Determinisme

Slice ini dijalankan tiga kali dengan hasil identik, termasuk daftar task ID yang lulus. Artinya selisih apa pun pada pengukuran berikutnya berasal dari perubahan checkpoint, bukan variasi pengukuran.

## Verifier lokal, bukan Docker

`research_assistant.coding_eval` memverifikasi solusi di dalam image Docker yang dipin. Docker tidak terpasang di mesin ini, dan pemasangannya memerlukan hak administrator serta WSL2, jadi pengukuran memakai `research_assistant.coding_bench` dengan verifier lokal.

Kedua verifier menjalankan program yang sama dengan pemilihan task, prompt, dan pengaturan sampling yang identik (`seed 917`, `temperature 0`, `max_tokens 768`). Bedanya pada isolasi:

| | Docker | Lokal |
| --- | --- | --- |
| Proses terpisah | ya | ya |
| Batas waktu | 12 s | 12 s |
| Isolasi jaringan | ya (`--network none`) | tidak |
| Batas memori/CPU | 192 MB, 1 CPU | tidak |
| Pengguna terpisah | ya | tidak |

Verifier lokal divalidasi dengan menjalankan 24 solusi referensi dari dataset: **24/24 lulus**, sama dengan yang dilaporkan untuk verifier Docker. Ini menunjukkan verifier tidak terlalu ketat maupun terlalu longgar pada kasus yang sudah diketahui benar.

Karena isolasinya lebih lemah, jangan memakai verifier lokal untuk kode yang tidak dipercaya. Di sini kode dihasilkan model lokal, dan itu diterima sebagai risiko yang disengaja.

## Perbandingan dengan angka dokumentasi

Angka di `docs/coding-upgrade.md` diukur pada Apple M1 Pro dengan verifier Docker. Angka di bawah diukur pada Windows dengan RTX 4060 Ti dan verifier lokal.

| Checkpoint | Dokumentasi (Docker, M1 Pro) | Mesin ini (lokal, CUDA) |
| --- | --- | --- |
| write-python | 10/24 | 11/24 |
| repair-one-tool-call | 5/12 | 4/12 |

Selisihnya satu soal bertukar arah di tiap slice:

- write-python: bertambah 287 dan 393, kehilangan 240.
- repair: kehilangan 223.

Selisih sekecil ini konsisten dengan perbedaan backend numerik (Metal versus CUDA) dan isolasi verifier. **Kedua kolom tidak boleh dibandingkan langsung** karena verifier dan perangkatnya berbeda. Untuk menilai perubahan model, bandingkan hanya di dalam kolom "mesin ini".

## Pola kegagalan yang perlu dicatat

Rincian penyebab gagal lebih informatif daripada skornya:

- write-python, 13 gagal: **9 tidak menghasilkan kode sama sekali** (74, 77, 87, 130, 162, 172, 259, 291, 306) dan 4 menghasilkan kode yang salah atau terpotong (71, 240, 295, 432).
- repair, 8 gagal: **5 tidak mengirim tool call `edit` yang sah** (74, 87, 162, 223, 226).

Jadi kegagalan terbanyak bukan soal kemampuan menulis kode, melainkan model menjawab dengan prosa penalaran alih-alih kode atau tool call. Ini sejalan dengan varian "Python prefill" di dokumentasi yang menaikkan skor dari 10 ke 12 pada mesin mereka: memaksa awalan blok kode mengurangi kegagalan bentuk ini.

Konsekuensinya, perbandingan checkpoint sebaiknya mencatat proporsi kegagalan "tidak menghasilkan artefak" secara terpisah, karena perbaikan pada format jawaban bisa menaikkan skor tanpa perubahan kemampuan kode.

## Mengulang dan membandingkan

```powershell
# verifikasi verifier
python -m research_assistant.coding_bench --selfcheck

# ukur state saat ini
python -m research_assistant.coding_bench --checkpoint nama-checkpoint --tasks both

# ukur dengan cara elicitation tertentu
python -m research_assistant.coding_bench --checkpoint nama --elicit adaptive --tasks both

# ukur checkpoint dengan adapter LoRA
python -m research_assistant.coding_bench --checkpoint nama-lora --lora runs\adapter.gguf

# bandingkan dua checkpoint
python -m research_assistant.coding_bench --compare lfm3-base-20261119 nama-lora
```

Nilai `--elicit`:

| Nilai | Perilaku | Hasil terukur |
| --- | --- | --- |
| `plain` | instruksi chat saja (baseline) | 11/24, 4/12 |
| `strict` | prefill kode + schema JSON dipaksa | 12/24, **3/12** |
| `adaptive` | prefill kode + retry bersyarat | 12/24, **7/12** |

`strict` sengaja disimpan sebagai pembanding: memaksa schema penuh pada tool call terbukti menurunkan hasil, dan itu konsisten dengan temuan di [solusi format](format-solution.md).

`--exec-rounds N` menambahkan putaran perbaikan berbasis eksekusi pada write-python, memakai contoh uji yang terlihat di prompt. Hasilnya 13/24 pada satu putaran dan **14/24** pada dua putaran, lalu jenuh. Rinciannya di [solusi format](format-solution.md#putaran-perbaikan-berbasis-eksekusi).

Dataset berada di `.runtime/coding-eval/sanitized-mbpp.json` dengan SHA-256 yang dipin, dan pemilihannya diverifikasi sama persis dengan daftar task di `docs/benchmark-results.json`.

## Catatan

- Satu run penuh (24 + 12 task) memerlukan sekitar tujuh menit pada mesin ini.
- `temp=0` tidak menjamin determinisme lintas perangkat keras, jadi jangan membandingkan hasil CUDA dengan hasil Metal seolah satu deret.
- Skor ini mengukur checkpoint apa adanya, bukan mutu akhir produk. Angka 11/24 dan 4/12 adalah dasar pembanding, bukan target.
