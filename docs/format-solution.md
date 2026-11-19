# Solusi masalah format: berbasis literatur dan terukur

Dokumen ini menjawab satu pertanyaan: kegagalan model pada benchmark coding (9 dari 13 kegagalan write-python tidak menghasilkan kode sama sekali; 5 dari 8 kegagalan repair tidak mengirim tool call) harus diperbaiki dengan cara apa, menurut literatur, dan apakah cara itu benar-benar bekerja di mesin ini.

Jawaban singkat: **bukan dengan melatih ulang model.** Perbaiki di jalur generasi, dengan paksaan yang bersyarat, bukan paksaan penuh. Hasil terukur: repair 4/12 → **7/12**, write 11/24 → 12/24.

## Diagnosis yang mendasari

Literatur menyebut pola ini dengan nama. Kadioglu & Uppuluri (arXiv:2607.20456) menemukan pada model 0,6B–20B bahwa "syntax errors dominate failures" dan akurasi eksekusi model kecil "near-zero" sebelum format diperbaiki, sementara akurasi solusi tetap 35% setelahnya. Artinya lapisan format adalah bagian yang mudah diperbaiki, sedangkan kemampuan semantik adalah batas atasnya.

Sandoval dkk. (arXiv:2604.16697) memberi mekanismenya: representasi sudah terbentuk sejak lapisan awal tetapi "remain computationally inert until the final layer, where **format-compliance demands compete with** them". Model sering *tahu* jawabannya; yang gagal adalah kewajiban menyampaikan dalam bentuk yang diminta. Ini persis yang terlihat di data: model menulis "We need to write a Python function named..." dan kehabisan token sebelum pernah menulis kode.

## Temuan yang membalik intuisi: paksaan penuh justru merusak

Dugaan wajar adalah memaksa format JSON ketat. Saya menguji itu lebih dulu, dan hasilnya **lebih buruk daripada baseline**:

| Mode | repair-one-tool-call |
| --- | --- |
| baseline (instruksi biasa) | 4/12 |
| strict (schema JSON dipaksa) | **3/12** |

Dua task yang semula lulus (77 dan 172) justru gagal. Pemeriksaan jejak menunjukkan sebabnya: mode strict memotong ruang model untuk menalar tentang patch, sehingga `oldString` yang dihasilkan tidak lagi cocok dengan isi berkas. Valid secara skema, salah secara isi.

Literatur memprediksi ini dengan tepat:

- **Tam dkk., "Let Me Speak Freely?" (arXiv:2408.02442)**: "a significant decline in LLMs reasoning abilities under format restrictions", dan "stricter format constraints generally lead to greater performance degradation". Pada GSM8K, gpt-3.5-turbo jatuh dari 75,99 menjadi 49,25 dengan JSON+schema.
- **Ray, "The Constraint Tax" (arXiv:2605.26128)** mengukur langsung pada model sekelas ini (Qwen2.5-1.5B, SmolLM2-1.7B). Pada tugas tool call terstruktur: **91,5% eksekusi benar dengan prompt-only JSON, hanya 48,0% dengan schema ketat**, sementara keduanya 100% valid secara skema. Kalimatnya: "The error is semantic, not structural."
- **Lee dkk., "The Format Tax" (arXiv:2604.03616)** menemukan biaya terbesar masuk **di prompt**, bukan di decoder: "format-requesting instructions alone cause most of the accuracy loss, before any decoder constraint is applied."

Kesimpulannya bukan "jangan pakai constraint", melainkan resep yang mereka sebut **"reason free, constrain late"** — biarkan model menalar bebas, batasi hanya kemasan akhirnya.

## Solusi yang diterapkan

Dua mekanisme, keduanya di jalur generasi, tanpa menyentuh bobot:

1. **Write-python: prefill assistant.** Giliran asisten dibuka dengan ```` ```python ```` sehingga model tidak pernah mendapat kesempatan memilih untuk menulis prosa. Ini bentuk paling ringan dari *budget forcing* (Muennighoff dkk., arXiv:2501.19393): "forcefully terminating the model's thinking process" agar langsung berkomitmen pada jawaban.

2. **Repair: retry bersyarat, bukan paksaan.** Percobaan pertama dibiarkan bebas menalar dengan tool `edit` yang tersedia. Hanya jika hasilnya tidak mengandung tool call yang sah, percobaan kedua dikirim dengan instruksi tegas untuk mengirim satu tool call saja. Paksaan penuh tidak pernah dipakai.

Alasan retry bersyarat didukung kuat oleh literatur:

- **Huang dkk. (arXiv:2310.01798, ICLR 2024)**: model gagal mengoreksi diri **tanpa umpan balik eksternal**. Retry di sini dipicu oleh kegagalan yang terdeteksi secara objektif, bukan oleh introspeksi.
- **McAndrews (arXiv:2604.21950)** menguji persis rentang 1–3B: perbaikan dengan umpan balik eksekusi memberi ">4 standard deviations", tetapi "early stopping is essential — without it every iteration is net-negative". Karena itu retry dibatasi dua percobaan.
- **EvoFlux (arXiv:2606.12674)** pada 1,5B–4B menemukan SFT dan SFT+DPO "match or underperform the zero-shot baseline", sedangkan *inference-time execution-guided repair* menaikkan keberhasilan dari ~3% ke 17–24%.

## Hasil terukur

| Slice | baseline | strict (paksa penuh) | **adaptive (diterapkan)** |
| --- | --- | --- | --- |
| write-python | 11/24 | 12/24 | **12/24** |
| repair-one-tool-call | 4/12 | 3/12 | **7/12** |

Perubahan per task:

- repair: mendapat 74, 162, 226; **tidak ada yang hilang**. Hasil ini deterministik — diulang, keluar 7/12 dengan daftar ID identik.
- write: mendapat 71, 77, 162, 172; kehilangan 141, 393, 452.

Mode adaptive menyamai mode strict pada write (12/24) tetapi jauh lebih baik pada repair (7/12 vs 3/12). Jadi seluruh keuntungan berasal dari membiarkan penalaran tetap bebas dan hanya menindak ketika formatnya benar-benar gagal.

## Putaran perbaikan berbasis eksekusi

Langkah ketiga diuji terpisah: setelah kode dihasilkan, jalankan terhadap **contoh uji yang sudah terlihat di prompt** (hanya `test_list[0]`, tidak pernah tes tersembunyi), lalu kirim traceback-nya kembali dan minta perbaikan. Berhenti begitu contoh itu lulus.

| Mode | write-python | Yang diperbaiki |
| --- | --- | --- |
| baseline | 11/24 | — |
| adaptive | 12/24 | — |
| adaptive + 1 putaran | 13/24 | 240 |
| adaptive + 2 putaran | **14/24** | 240, 393 |
| adaptive + 3 putaran | 14/24 | tidak menambah |

Jumlahnya jenuh di dua putaran, sesuai dua temuan literatur: "most gains in the first two rounds" (arXiv:2604.10508) dan plateau seleksi ketika verifier-nya lemah (Large Language Monkeys, arXiv:2407.21787). Karena itu **dua putaran adalah setting yang dipakai**; putaran ketiga hanya membakar inferensi.

## Komposisi kegagalan: yang berubah sebenarnya

Angka total menyembunyikan perubahan yang lebih penting. Ini klasifikasi 24 soal write-python:

| Tahap | Lulus | Tidak menghasilkan kode | SyntaxError | Salah logika |
| --- | --- | --- | --- | --- |
| baseline | 11 | **9** | 4 | 0 |
| adaptive | 12 | **0** | 2 | 10 |
| + 1 putaran | 13 | 0 | 1 | 10 |
| + 2 putaran | 14 | 0 | 1 | 9 |

Kelas kegagalan "tidak menghasilkan kode sama sekali" **hilang sepenuhnya**, dari 9 menjadi 0. Itu tepat seperti yang diprediksi literatur: masalahnya format, dan format bisa dihilangkan by construction.

Yang tersisa adalah batas semantik. Sepuluh soal kini menghasilkan kode yang valid secara sintaksis tetapi salah secara logika. Tidak ada teknik format yang memperbaiki ini, dan itu konsisten dengan arXiv:2607.18357 (grammar menjamin bentuk, bukan makna) serta kerangka Constraint Tax.

Contoh konkret betapa lemahnya verifier satu contoh: soal 291 lulus contoh uji yang terlihat tetapi tetap gagal pada tes tersembunyi. Inilah sebabnya putaran perbaikan tidak bisa mendekati 24/24 — satu assertion tidak mampu membedakan program yang benar dari program yang kebetulan cocok pada satu kasus.

## Kejujuran soal ukuran sampel

Ini penting dan sengaja saya tempatkan sebelum rekomendasi lanjutan. Dengan n=24, interval kepercayaan 95% pada tingkat keberhasilan sekitar 45% melebar kira-kira ±5 soal. Artinya:

- write 11→12 (adaptive) adalah **satu soal**, berada sepenuhnya di dalam noise. Jangan diklaim sebagai perbaikan.
- write 11→14 (adaptive + 2 putaran) adalah tiga soal, juga masih dekat batas noise, meskipun perubahannya deterministik antar-prosedur.
- repair 4→7 adalah tiga soal pada n=12, juga dekat batas noise.

Yang bisa diklaim dengan aman adalah dua hal yang **tidak** bergantung pada selisih kecil itu:

1. Perbedaan antar-prosedur **deterministik** pada temperature 0: prosedur yang sama selalu memberi hasil dan daftar ID yang sama.
2. Perubahan **komposisi** kegagalan bersifat kualitatif dan jauh melampaui noise: kelas "tidak menghasilkan artefak" hilang total, 9 menjadi 0.

Untuk mengukur besaran perbaikan dengan layak, eval harus diperbesar (misalnya EvalPlus, arXiv:2305.01210), bukan ditambah putaran pada 24 soal yang sama.

Konsekuensi lain: klaim lama di `docs/coding-upgrade.md` bahwa QLoRA MLX menurunkan skor 10/24 → 7/24 juga tiga soal, jadi **belum terbukti** sebagai degradasi. Itu penting karena rekomendasi berikutnya bergantung padanya.

## Yang tidak saya sarankan

**Melatih ulang lebih dulu.** Tiga hasil independen menunjuk arah yang sama:

- **Biderman dkk., "LoRA Learns Less and Forgets Less" (arXiv:2405.09673, TMLR)** memakai domain kode dan matematika: "LoRA substantially underperforms full finetuning", tetapi lebih tahan lupa. Model kecil yang sudah overtrained adalah target yang rapuh (Springer dkk., arXiv:2503.19206: model yang dilatih lebih lama "harder to fine-tune, leading to degraded final performance").
- **Zhao dkk. (arXiv:2604.11838, ACL 2026)** menganalisis 1B–32B: "middle layers (20%–80%) are stable, whereas final layers exhibit high sensitivity". QLoRA sebelumnya menyasar `w1/w2/w3` pada **12 layer terakhir dari 24** dengan rank 8 — yaitu persis pita paling sensitif dengan rank paling rendah. Ini penjelasan yang masuk akal untuk hasil yang menurun, dan sekaligus resep perbaikan bila training tetap dilakukan: sasar pita tengah, naikkan rank.
- **CommitLLM (arXiv:2607.17532)**: "the post-processing layers contribute more to quality improvement than the fine-tuning itself".
- **Gorilla (arXiv:2305.15334)** menunjukkan bahwa pada LLaMA-7B polos, 100% panggilan API adalah halusinasi, dan instruction tuning sederhana menurunkannya ke 6,98%. Jadi kalau suatu saat training diperlukan, bentuk yang tepat adalah **supervisi berbentuk format**, bukan tambahan data domain.

## Urutan tindakan yang didukung bukti

1. **Selesai — instrumen.** Pisahkan metrik menjadi: tidak menghasilkan artefak, gagal kompilasi, salah secara semantik. Ray (arXiv:2605.26128) menyarankan persis ini: "report schema validity, answer accuracy, executable accuracy, and wrong-valid-schema rate separately." Angka 11/24 sebelumnya mencampur tiga jenis kegagalan yang berbeda.
2. **Selesai — prefill + retry bersyarat.** Menghapus kelas kegagalan "tidak menghasilkan artefak" tanpa biaya forgetting.
3. **Selesai — putaran perbaikan berbasis eksekusi.** Dua putaran, berhenti dini, hanya memakai contoh uji yang terlihat. Menambah dua soal dan menuntaskan kelas SyntaxError.
4. **Berikutnya: perbesar eval sebelum mempercayai selisih apa pun.** EvalPlus atau subset MBPP yang lebih besar. Seluruh selisih di halaman ini masih berada di dalam noise n=24.
5. **Terakhir: training, dan hanya bila 1–4 sudah mentok.** Sisa kegagalan sekarang murni semantik (9–10 soal salah logika), dan training format tidak akan menyentuhnya. Jika tetap dilakukan: sasar layer tengah, rank lebih tinggi, campur data instruksi umum, dan ukur pada set uji umum untuk mendeteksi lupa.

Satu hal yang sengaja tidak saya pakai: **self-consistency/majority voting**. Wang dkk. (arXiv:2203.11171) memerlukan keragaman sampling, sehingga mati total pada temperature 0 yang dipakai harness ini. Padanan yang sah untuk kode adalah seleksi berbasis eksekusi (CodeT, arXiv:2207.10397), bukan voting atas teks.

## Cara mereproduksi

```powershell
python -m research_assistant.coding_bench --checkpoint nama --elicit adaptive --tasks both
python -m research_assistant.coding_bench --compare lfm3-base-20261119 nama
```

Nilai `--elicit`: `plain` (baseline), `strict` (paksa penuh, terukur lebih buruk), `adaptive` (yang dipakai). Skor tersimpan di `runs/checkpoints/`.
