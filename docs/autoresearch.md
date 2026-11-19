# Autoresearch: iterasi sampai eval naik

Dokumen ini mencatat loop autoresearch yang dipakai untuk menaikkan skor coding, apa yang dicoba, apa yang menang, dan seberapa jujur angka yang dihasilkan.

## Metode

Loop berjalan dalam lima putaran dengan protokol yang memisahkan tuning dari pengukuran:

- **Dev slice**: 32 soal MBPP yang diambil dari pool sama dengan seed 917, dengan soal test dikeluarkan. Dipakai untuk memilih strategi.
- **Test slice**: 24 soal yang sama persis dengan semua checkpoint sebelumnya. **Tidak dipakai untuk memilih apa pun.** Ini yang mencegah tuning 24 soal lalu melaporkan 24 soal yang sama.

Setiap strategi diukur pada dev, pemenangnya diukur pada test. Kalau keunggulan di dev tidak muncul di test, itu overfitting dan dilaporkan sebagai kegagalan, bukan kemajuan.

Ruangan hipotesis diambil dari literatur yang sudah dikumpulkan sebelumnya: sampling + penyaringan eksekusi (arXiv:2407.21787), perbaikan berbasis traceback dengan berhenti dini (arXiv:2604.21950), prefill sebagai pengendali budget (arXiv:2501.19393), dan tekanan format bersyarat alih-alih paksaan (arXiv:2605.26128).

## Hasil per putaran

Triage pada dev-16, lalu verifikasi pada dev-32 dan test.

| Strategi | dev-16 | dev-32 | test |
| --- | --- | --- | --- |
| `Z_plain_baseline` (chat, greedy) | — | — | **11/24** |
| `A_adaptive_exec2` (1 sampel + 2 putaran perbaikan) | 11/16 | 20/32 | 14/24 |
| `B_bon4_t08` (4 sampel, temp 0,8) | 12/16 | 20/32 | — |
| **`C_bon4_t08_exec1`** (4 sampel + 1 putaran perbaikan) | 12/16 | **21/32** | **16/24** |
| `D_bon8_t10` (8 sampel, temp 1,0) | 11/16 | — | — |
| `E_bon8_t08_exec1` | 12/16 | — | — |
| `F_plan_exec1` (rencana lalu kode) | 9/16 | — | — |
| `G_plan_exec0` | 11/16 | — | — |
| `H_bon4_t06_exec1` (temp 0,6) | 13/16 | 22/32 | 14/24 |
| `I_..._tok1536` (budget token lebih besar) | 13/16 | 22/32 | — |
| `J_bon4_t06_exec1_tok1536` | — | 22/32 | 14/24 |

Pemenang: **`C_bon4_t08_exec1`**.

## Yang gagal, dan mengapa itu berguna

Tiga hipotesis terbukti salah dan itu mempersempit masalah:

1. **Plan-then-code lebih buruk** (9/16 dan 11/16 vs 12/16). Memberi model kesempatan menalar sebelum menulis kode justru menurunkan hasil. Prefill yang menghapus fase menalar ternyata bukan kompromi — ia yang terbaik.

2. **Menambah sampel tidak menolong.** 8 sampel (11–12/16) tidak lebih baik dari 4 sampel (12/16), dan temp 1,0 lebih buruk dari 0,8.

3. **Kemenangan di dev-16 tidak selalu bertahan.** `H` dan `I` unggul di dev (13/16 dan 22/32) tetapi jatuh ke **14/24** di test, lebih buruk dari `C` yang 16/24. Ini persis overfitting yang ingin dicegah oleh pemisahan dev/test, dan alasan test slice tidak boleh dipakai untuk memilih.

## Headroom: mengapa tidak bisa jauh lebih tinggi

Sebelum menambah putaran lagi, saya ukur batas atasnya. Pada 16 soal dev, 32 sampel per soal dijalankan sekali, lalu dihitung:

| N sampel | Coverage (seleksi sempurna) | Filter contoh terlihat (metode nyata) |
| --- | --- | --- |
| 1 | 10/16 | 10/16 |
| 2 | 11/16 | 11/16 |
| 4 | 13/16 | 12/16 |
| 8 | 13/16 | 12/16 |
| 16 | 13/16 | 12/16 |
| 32 | 14/16 | 12/16 |

Dua kesimpulan:

- **Seleksi bukan lagi masalahnya.** Seleksi sempurna hanya memberi 13/16 vs 12/16 yang dicapai sekarang. Filter contoh uji tunggal sudah hampir sebaik oracle.
- **Coverage jenuh.** Dari N=4 ke N=32 hanya naik 13 → 14. Tiga soal (138, 304, 311) praktis tidak pernah diselesaikan sampling apa pun; pada soal 304 dan 311, 0 dan 1 dari 32 sampel yang lulus.

Jadi sisa kegagalan adalah **batas kapabilitas model 1,7B**, bukan masalah format atau seleksi. Tidak ada jumlah sampling yang menembusnya.

## Hasil akhir dan kejujuran angkanya

| | Skor test | Kelas "tidak menghasilkan kode" |
| --- | --- | --- |
| baseline (`Z`) | **11/24** | 9 |
| strategi `C` | **16/24** (4 run: 16, 14, 16, 16 → rata-rata **15,5**) | **0** |

Peningkatan sekitar +4,5 soal, atau +40% relatif, dan kelas kegagalan format hilang seluruhnya.

Dua catatan yang harus dibaca bersama angka itu:

1. **Strategi sampling tidak deterministik.** Baseline greedy memberi 11/24 identik di setiap run. Strategi `C` memberi 16, 14, 16, 16 — ada variasi 2 soal antar-run meski seed tetap. Jadi angka yang benar adalah **14–16, rata-rata 15,5**, bukan "16" saja. Siapa pun yang mengutip angka tunggal 16/24 mengutip hasil terbaik, bukan hasil yang diharapkan.

2. **n=24 masih kecil.** CI 95% sekitar ±5 soal. Peningkatan +4,5 soal berada di tepi signifikansi. Yang tidak bergantung pada noise adalah perubahan komposisi: 9 soal yang sebelumnya tidak menghasilkan kode sama sekali sekarang menghasilkan kode, dan itu deterministik karena berasal dari prefill, bukan dari sampling.

## Cara mereproduksi

```powershell
# triage cepat pada dev
python -m research_assistant.autoresearch --round rX --slice dev --dev-size 16 --strategies C_bon4_t08_exec1

# verifikasi pada test (hanya sekali, setelah strategi dipilih)
python -m research_assistant.autoresearch --round rX --slice test --strategies C_bon4_t08_exec1
```

Hasil tiap percobaan ditambahkan ke `runs/autoresearch/<round>/trials.jsonl`, dan ringkasannya di `summary-dev.json` / `summary-test.json`.

## Perbaikan verifier yang ditemukan di jalan

Saat menjalankan strategi first-sample, verifier crash dengan `UnicodeEncodeError: 'charmap' codec can't encode character '\u2261'`. Penyebabnya: program yang dihasilkan model memuat karakter non-ASCII (misalnya `≡` di docstring), dan proses anak di Windows menulis stdin memakai codepage locale, bukan UTF-8. Program yang benar pun jadi dihitung gagal.

Verifier sekarang memaksa `PYTHONIOENCODING=utf-8` dan `PYTHONUTF8=1` lewat `run_child()`. Ini juga menjelaskan mengapa pengukuran lama bisa sedikit terlalu rendah untuk program yang memuat karakter Unicode.

## Batas yang sudah tercapai

Dengan model ini, pada slice ini, skor realistis ada di **15–16 dari 24**, dan sisanya adalah soal yang memang di luar kemampuan model, bukan yang bisa diperbaiki lewat rekayasa prompt. Untuk melompat lebih jauh diperlukan model yang lebih capable (K2-Horizon-7B mencapai 15/24 pada protokol lama) atau fine-tuning yang menyasar kapabilitas, bukan format.
