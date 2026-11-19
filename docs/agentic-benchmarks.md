# Benchmark agentic coding

Versi publik terbaru yang ditemukan pada 18 November 2026 adalah **Terminal-Bench 4.1.0** (66 tugas) dan **DeepSWE 1.1** (113 tugas). Skor 12,92 pada model card berasal dari Terminal-Bench 2.1 dengan OpenCode dan checkpoint BF16. Skor SWE-bench di sana memakai set tugas SWE-bench Verified dan Multilingual. Gunakan baseline baru dari GGUF lokal untuk mengukur efek patch. [Rilis Terminal-Bench](https://github.com/harbor-framework/terminal-bench/releases/tag/v4.1.0) · [DeepSWE](https://deepswe.datacurve.ai/) · [Model card](https://huggingface.co/LiquidAI/LFM3-1.7B#benchmarks)

## Menjalankan baseline dan patch

Jalankan dari root proyek dengan Docker Desktop aktif. `setup` menyiapkan Harbor 0.26.0 dan checkout revisi benchmark yang dipin. Runner memverifikasi checksum paket npm OpenCode 1.19.6 saat mengisi cache biner Linux. Harbor menyalin biner yang sesuai arsitektur ke container saat tahap setup. Benchmark memakai llama.cpp lokal melalui loopback Docker Desktop, 32K konteks secara default, Thinking aktif, satu percobaan per tugas, dan satu trial pada satu waktu. Uji singkat membatasi waktu agen ke 270 detik per tugas. Pemasangan agen, build image, dan verifikasi memiliki waktu terpisah.

```sh
./wsp setup
./wsp bench setup
./wsp bench both --smoke
./wsp bench both --agent-seconds 900 --context-tokens 65536
./wsp bench both --smoke --experiment runs/weight-space/NAMA-EKSPERIMEN
```

Mode `--smoke` mengambil satu tugas resmi dari masing-masing suite. Tanpa opsi itu, runner mengambil dua tugas per suite, yaitu perbaikan pemrosesan sesi dan pemulihan WAL untuk Terminal-Bench, serta pemeriksa referensi dokumentasi Go dan fitur daftar isi Obsidian untuk DeepSWE. Subset ini bisa dijalankan di Mac tanpa GPU NVIDIA. Harbor menjalankan verifier dari rilis masing-masing dan DeepSWE memakai container verifier terpisah. Akses jaringan trial untuk inferensi lokal diberikan melalui allowlist Harbor.

Laporan `benchmark.json` di `runs/agentic-bench/` menyimpan versi, hash checkpoint, hasil per tugas, pemakaian token, jumlah panggilan tool, dan lokasi jejak Harbor. Hitungan sebelum tool edit pertama mencakup panggilan yang tercatat sebelum `edit`, `write`, atau `apply_patch`. Perubahan file lewat Bash hanya terlihat di jejak. `error` menandai trial tanpa skor verifier yang sah. Dalam mode `--experiment`, runner memakai tugas dan konfigurasi yang sama untuk baseline, checkpoint edit, dan kontrol F32 jika ada. Hasilnya berlaku untuk subset, waktu, dan runtime tersebut. [Format tugas Harbor](https://www.harborframework.com/docs/tasks) · [DeepSWE v1.1](https://github.com/datacurve-ai/deep-swe)

Uji integrasi pada 18 November menghasilkan skor verifier 0 pada masing-masing tugas smoke setelah agen mencapai batas 270 detik. Trial awal sempat berjalan bersamaan. Log DeepSWE juga merekam permintaan 49.052 token yang melampaui konteks 32K. Untuk perbandingan patch yang memakai waktu sebagai batas, jalankan baseline dan kandidat berurutan, tanpa beban inferensi lain pada perangkat. Opsi `--reasoning-budget` membatasi token thinking per respons di runtime dan dicatat di laporan agar bisa direproduksi.

`--edit-test-workflow` memberi OpenCode instruksi umum untuk membuat perubahan kecil, menjalankan tes yang relevan, lalu memakai hasilnya untuk langkah berikutnya. Teks instruksi ikut disimpan di laporan. Gunakan pengaturan runtime dan workflow yang sama saat mengukur efek patch bobot.

Untuk perubahan bobot, gunakan tugas pengembangan terpisah untuk mereproduksi kegagalan yang terlihat di trajectory. Bekukan kandidat patch sebelum evaluasi pada tugas yang disisihkan, dan ukur hasil verifier serta kegagalan per tugas.

## Hasil smoke test 64K

Setiap baris mencakup satu tugas resmi, konteks 65.536 token, OpenCode 1.19.6, dan budget agen yang dikonfigurasi 420 detik. Waktu yang tercatat oleh Harbor pada beberapa trial melebihi budget tersebut.

| Suite | Thinking tanpa batas eksplisit | Thinking 512 token | Thinking 512 dan workflow edit–test |
| --- | --- | --- | --- |
| Terminal-Bench 4.1.0 | 0/1 | 0/1 | Belum diuji |
| DeepSWE 1.1 | 0/1 | 0/1 | 0/1 |

Batas 512 token mengurangi panjang respons pertama DeepSWE dari 3.405 menjadi 525 token. Panggilan tool edit pertama muncul setelah 19 panggilan pada Terminal-Bench dan 55 pada DeepSWE. Pada uji workflow edit–test, OpenCode menjalankan 37 panggilan Bash dan baca tanpa panggilan tool edit. Skor DeepSWE tetap 0 dari 3 tes fitur dan 16 dari 16 tes regresi pada ketiga konfigurasi.

Patch WSP satu baris Q6_K pada token penutup thinking menghasilkan 3/4 contoh sintetis yang benar dan median 882,5 token, sama dengan baseline pada set itu. Perubahan skor benchmark dari patch bobot belum terverifikasi. [Data hasil benchmark dan probe](benchmark-results.json)

Eksperimen QLoRA, LFM3 QAD, dan checkpoint K2-Horizon berikutnya dicatat di [evaluasi model coding lokal](coding-upgrade.md).
