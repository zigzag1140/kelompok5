# WiFi CSI-Based Human Activity Recognition on UT-HAR Dataset Using 1D-CNN and Bi-LSTM Attention

**Proyek Pengganti Ujian Tengah Semester (UTS) — Mata Kuliah Pembelajaran Mesin Lanjut (Track A)**  
**Departemen Teknik Komputer, Fakultas Teknologi Informasi, Universitas Andalas**

---

## Kelompok 5
| No | Nama Mahasiswa | NIM | 
| :-: | :--- | :---: | 
| 1 | **Muhammad Al Fath** | 2311511013 |
| 2 | **Muhammad Rizki Hapiz** | 2311511017 | 
| 3 | **Rahmadatul Afdal** | 2311512021 | 
| 4 | **Naufal Rafiif Irwan** | 2311512025 | 

---

## 1. Latar Belakang & Deskripsi Proyek
Pengenalan aktivitas manusia di dalam ruangan memiliki peran krusial dalam aplikasi *smart home*, pemantauan kesehatan lansia, dan sistem deteksi jatuh (*fall detection*). Pendekatan konvensional berbasis kamera terkendala oleh masalah privasi penghuni, pencahayaan ruangan, dan halangan objek, sementara  *wearable sensors* sering kali tidak nyaman atau lupa dikenakan.

Proyek ini mengimplementasikan sistem **HAR tanpa sensor melekat** dengan memanfaatkan sinyal **WiFi *Channel State Information* (CSI)**. Ketika tubuh manusia bergerak di antara antena *transmitter* dan penerima *receiver*, pantulan gelombang radio dan pergeseran frekuensi Doppler menciptakan pola perubahan amplitudo yang unik pada setiap *subcarrier* frekuensi.

Dalam repositori ini, kami merancang, melatih, dan membandingkan dua arsitektur *Deep Learning* berbasis PyTorch pada dataset publik **UT-HAR**:
1. **Baseline Model:** *1D-Convolutional Neural Network* (`Baseline1DCNN`).
2. **Proposed Model:** *Hybrid 1D-CNN + Bi-directional Long Short-Term Memory dengan Mekanisme Temporal Attention* (`ProposedCNNBiLSTM`).

---

## 2. Identitas & Spesifikasi Teknis Dataset (UT-HAR)
Dataset yang digunakan adalah **UT-HAR** (*University of Toronto Human Activity Recognition*), salah satu *dataset benchmark* utama pada repositori rujukan [Awesome-WiFi-CSI-Sensing](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) (Yousefi et al., *IEEE Communications Magazine*, 2017).

* **Perangkat Keras Akuisisi:** Intel 5300 WiFi *Network Interface Card* (NIC) dengan *Linux 802.11n CSI Tool*.
* **Konfigurasi Antena (MIMO):** $N_t = 1$ antena pemancar (*transmitter*) dan $N_r = 3$ antena penerima (*receiver*).
* **Jumlah Subcarrier Frekuensi:** Setiap pasangan lintasan antena mengekstrak $K = 30$ grup *subcarrier* OFDM. Dengan demikian, total kanal informasi per paket WiFi adalah:
  $$C = N_t \times N_r \times K = 1 \times 3 \times 30 = 90 \text{ kanal subcarrier}$$
* **Dimensi *Time-Steps*:** Setiap sampel aktivitas disegmentasi ke dalam jendela waktu tetap sepanjang $T = 250$ paket berurutan, sehingga setiap sampel direpresentasikan sebagai matriks 2D berukuran **$250 \times 90$** (atau tensor $\mathbb{R}^{90 \times 250}$ pada input lapisan `Conv1d`).
* **Kelas Target (4 Kelas Aktivitas Utama):**
  * `0` : **Walk** (Berjalan)
  * `1` : **Sit** (Duduk)
  * `2` : **Stand** (Berdiri)
  * `3` : **Fall** (Jatuh)

> **Catatan Ketersediaan Dataset:** Karena ukuran file mentah `UT_HAR.zip` cukup besar, file dataset penuh tidak disertakan langsung di dalam repositori GitHub ini, melainkan diwakili oleh 4 sampel uji asli pada folder `demo_samples/` untuk pengujian inferensi lokal. Dataset asli dapat diunduh melalui repositori resmi [WiFi-CSI-Sensing-Benchmark](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark).

---

## 3. Metodologi Partisi Data & Mitigasi *Data Leakage*
Mengacu pada studi analisis kritis oleh **Varga (2024)** mengenai kebocoran data (*data leakage*) pada *benchmark* WiFi CSI HAR, pembagian sampel *sliding window* secara acak menyebabkan potongan sinyal yang bersebelahan secara waktu masuk ke dalam partisi *training* dan *testing* sekaligus, sehingga menyebabkan akurasi yang tidak realistis (mendekati $99\%–100\%$).

Untuk menjamin integritas evaluasi, proyek ini menerapkan protokol **Contiguous Block Partitioning dengan Guard Gap**:
1. **Pemisahan Blok (70% / 15% / 15%):** Rekaman sampel untuk setiap kelas dibagi secara berurutan berdasarkan blok waktu menjadi **70% Training**, **15% Validation**, dan **15% Testing**.
2. **Jeda Pembatas (*Guard Gap* $G = 8$ Sampel):** Sebanyak 8 sampel perbatasan di antara blok *Train–Validation* dan *Validation–Test* dibuang sepenuhnya untuk memutus korelasi *sliding window* antar-partisi.
3. **Isolasi Fungsi Partisi secara Ketat:**
   * **Training Set (1.795 sampel):** Digunakan murni untuk memperbarui bobot gradien model dan menghitung parameter normalisasi ($\mu_{\text{train}}, \sigma_{\text{train}}$).
   * **Validation Set (326 sampel):** Digunakan untuk memantau konvergensi di setiap *epoch*, melakukan studi ablasi (*ablation study*), dan memilih *checkpoint* bobot model dengan *Validation Loss* terendah (`best_val_loss`).
   * **Held-Out Test Set (363 sampel):** Diisolasi sepenuhnya selama pelatihan dan hanya dieksekusi **satu kali (*single-shot evaluation*)** di akhir eksperimen.

### Tabel Distribusi Sampel per Partisi (UT-HAR 4 Kelas)
| Partisi Dataset | Proporsi Blok | Walk (0) | Sit (1) | Stand (2) | Fall (3) | Total Sampel | Fungsi Metodologis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training Set** | 70% Blok Awal | 1.017 | 272 | 204 | 302 | **1.795** | Optimasi bobot & perhitungan ($\mu_{\text{train}}, \sigma_{\text{train}}$) |
| **Validation Set** | 15% Blok Tengah | 203 | 44 | 29 | 50 | **326** | Seleksi *checkpoint epoch* terbaik & studi ablasi |
| **Held-Out Test Set** | 15% Blok Akhir | 213 | 52 | 39 | 59 | **363** | Evaluasi akhir satu kali (*single-shot evaluation*) |

---

## 4. Tahapan Preprocessing Sinyal CSI
Sinyal WiFi CSI di lingkungan nyata sangat rentan terhadap *Additive White Gaussian Noise* / AWGN dan fluktuasi *multipath fading*. Tahapan *preprocessing* yang diterapkan pada proyek ini meliputi:

1. **Simulasi Gangguan Kanal Realistis:** Menambahkan perturbasi *multipath slow-fading envelope* dan derau kanal untuk menguji ketahanan filter serta kemampuan generalisasi model pada kondisi kanal nirkabel yang dinamis.
2. **Penyaringan *4th-Order Butterworth Low-Pass Filter* (Zero-Phase):**
   Meredam derau frekuensi tinggi sepanjang sumbu waktu ($T = 250$) menggunakan frekuensi *cutoff* ternormalisasi $W_n = \omega_c / \omega_{\text{Nyq}} = 0.16$ melalui filter maju-mundur (`scipy.signal.filtfilt`) agar tidak menimbulkan pergeseran fasa ($\Delta\phi(\omega) = 0$):
   $$|H_{LP}(\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{W_n \cdot \omega_{\text{Nyq}}}\right)^{2N}}, \quad N = 4$$
3. **Normalisasi Z-Score Bebas Kebocoran (*Zero-Leakage Standardization*):**
   Menghitung rata-rata $\mu_{\text{train}}$ dan simpangan baku $\sigma_{\text{train}}$ **hanya dari partisi Training**, lalu menormalisasi dan mentransposisikan setiap matriks sampel menjadi tensor $\mathbf{Z}_n \in \mathbb{R}^{90 \times 250}$:
   $$\mathbf{Z}_{n} = \left( \frac{\tilde{\mathbf{X}}_n - \mu_{\text{train}}}{\sigma_{\text{train}} + 10^{-8}} \right)^{\top}$$
4. **Analisis Spektral *Short-Time Fourier Transform* (STFT):**
   Menghapus komponen *DC offset* ($\bar{x}_c[t] = x_c[t] - \mu_x$) sebelum menerapkan transformasi STFT (jendela Hann $M = 32$, *overlap* $24$) untuk memvisualisasikan konsentrasi energi frekuensi Doppler selama aktivitas berlangsung.

---

## 5. Arsitektur Deep Learning yang Dibandingkan

### A. Model 1: `Baseline1DCNN` (27.140 Parameter)
Dirancang mengacu pada arsitektur pembanding *1D-CNN* di literatur Moshiri et al. (2021) untuk mengekstrak fitur lokal secara paralel:
* **Blok Konvolusi 1:** `Conv1d(90 -> 32, kernel_size=5, padding=2)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(2)` $\rightarrow$ `BatchNorm1d(32)` $\rightarrow$ `Dropout(0.3)`.
* **Blok Konvolusi 2:** `Conv1d(32 -> 64, kernel_size=5, padding=2)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(2)` $\rightarrow$ `BatchNorm1d(64)` $\rightarrow$ `Dropout(0.3)`.
* **Global Pooling & Classifier:** `AdaptiveAvgPool1d(1)` $\rightarrow$ `Flatten` $\rightarrow$ `Linear(64 -> 32)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(32 -> 4)`.
* **Keterbatasan:** Operasi `AdaptiveAvgPool1d(1)` melebur seluruh urutan waktu menjadi satu nilai rata-rata, sehingga informasi transisi temporal dari awal hingga akhir gerakan hilang.

### B. Model 2: `ProposedCNNBiLSTM` dengan Temporal Attention (261.445 Parameter)
Arsitektur hibrida yang menggabungkan ekstraksi fitur lokal konvolusional dengan pemodelan sekuensial dua arah dan mekanisme fokus perhatian waktu:
* **Spasial-Lokal CNN Feature Extractor:**
  * Blok 1: `Conv1d(90 -> 64, kernel_size=5, padding=2)` $\rightarrow$ `BatchNorm1d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(2)` $\rightarrow$ `Dropout(0.2)`.
  * Blok 2: `Conv1d(64 -> 128, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm1d(128)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1d(2)` $\rightarrow$ `Dropout(0.2)`.
  * Permutasi dimensi tensor dari $(B, 128, 62)$ menjadi urutan waktu $(B, T'=62, 128)$.
* **2-Layer Bi-directional LSTM (Bi-LSTM):**
  Memproses urutan fitur sepanjang $T'=62$ langkah waktu dari arah maju ($\overrightarrow{\mathbf{h}}_t$) dan arah mundur ($\overleftarrow{\mathbf{h}}_t$) dengan `hidden_size=64` per arah (2 lapisan, `dropout=0.2`), menghasilkan representasi kontekstual dua arah $\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t \parallel \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{128}$.
* **Temporal Attention Layer:**
  Menghitung bobot kepentingan $\alpha_t$ untuk setiap langkah waktu agar jaringan fokus pada segmen gerakan paling informatif dan mengabaikan jeda diam di awal/akhir jendela:
  $$e_t = \mathbf{w}_a^{\top} \tanh(\mathbf{W}_a \mathbf{h}_t + \mathbf{b}_a), \quad \alpha_t = \frac{\exp(e_t)}{\sum_{\tau=1}^{62} \exp(e_\tau)}, \quad \mathbf{v} = \sum_{t=1}^{62} \alpha_t \mathbf{h}_t$$
* **Classifier Head:** `Dropout(0.3)` $\rightarrow$ `Linear(128 -> 4)`.

---

## 6. Hasil Eksperimen & Analisis Performa

Seluruh model dilatih selama **50 Epoch** menggunakan *optimizer* **Adam** ($\text{lr} = 0.001, \beta_1 = 0.9, \beta_2 = 0.99, \epsilon = 10^{-9}$), ukuran *batch* 64, dan fungsi kerugian *Cross-Entropy*.

### A. Hasil *Ablation Study* pada Validation Set
Studi ablasi dilakukan pada `val_loader` (326 sampel) untuk menguji pengaruh *Butterworth Low-pass Filter* ($W_n = 0.16$) terhadap sinyal tanpa filter (*Raw CSI*):

| Konfigurasi Eksperimen / Varian Model | Jumlah Parameter | Best Epoch | Best Val Loss | Val Accuracy (%) | Val Macro-F1 (%) | Waktu Training |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Baseline 1D-CNN (Raw CSI — Tanpa Filter) | 27.140 | Epoch 48 | 0.0427 | 97.55% | 95.59% | 12.47 s |
| 2. Baseline 1D-CNN (Butterworth Filtered $W_n=0.16$) | 27.140 | Epoch 46 | 0.0720 | 96.32% | 93.60% | 11.92 s |
| 3. Proposed CNN-BiLSTM + Attention (Raw CSI — Tanpa Filter) | 261.445 | Epoch 37 | 0.1096 | 96.63% | 93.75% | 15.92 s |
| **4. Proposed CNN-BiLSTM + Attention (Filtered $W_n=0.16$)** | **261.445** | **Epoch 49** | **0.0809** | **96.01%** | **93.32%** | **15.90 s** |

> **Temuan Studi Ablasi:** Pada arsitektur **Proposed CNN-BiLSTM**, penerapan *Butterworth Low-pass Filter* ($W_n = 0.16$) terbukti menurunkan *Best Validation Loss* secara signifikan sebesar **26,18%** (dari `0.1096` menjadi **`0.0809`**). Peredaman derau frekuensi tinggi sangat membantu gerbang sekuensial Bi-LSTM dan *Temporal Attention* menghindari *overfitting* terhadap derau kanal.

---

### B. *Single-Shot Evaluation* pada Held-Out Test Set
Setelah bobot *checkpoint* terbaik dari tahap validasi dikunci, evaluasi akhir dijalankan tepat satu kali pada **Held-Out Test Set (363 sampel)**:

| Model (Best Validation Checkpoint) | Test Loss | Test Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1D-CNN** (Checkpoint Epoch 46) | 0.1347 | 95.04% | 91.98% | **92.94%** | 92.32% |
| **Proposed 1D-CNN + Bi-LSTM (Attention)** (Checkpoint Epoch 49) | **0.1168** | **95.87%** | **92.47%** | 92.92% | **92.68%** |

### C. Rincian Metrik Per-Kelas pada Test Set (Proposed Model)
| Kelas Aktivitas | Precision | Recall | F1-Score | Jumlah Sampel Uji (*Support*) | Analisis Performa |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Walk (Berjalan)** | 99.53% | 98.59% | **99.06%** | 213 | 210 dari 213 sampel berhasil diklasifikasikan dengan tepat. |
| **Sit (Duduk)** | 86.79% | 88.46% | 87.62% | 52 | 46 tepat; gerakan vertikal stasioner beririsan tipis dengan *Stand*. |
| **Stand (Berdiri)** | 86.84% | 84.62% | 85.71% | 39 | 33 tepat, 6 sampel terklasifikasi sebagai *Sit* akibat kemiripan pantulan. |
| **Fall (Jatuh)** | 96.72% | **100.00%** | **98.33%** | 59 | **Seluruh 59 kejadian jatuh terdeteksi 100% (*zero false negative*).** |
| **Macro Average** | **92.47%** | **92.92%** | **92.68%** | **363** | Rata-rata bobot setara lintas keempat kelas aktivitas. |
| **Weighted Average** | **95.88%** | **95.87%** | **95.87%** | **363** | Rata-rata tertimbang proporsional terhadap jumlah sampel. |

---

## 7. Struktur File & Penjelasan Tiap Komponen di Repositori

```text
wifi-csi-har/
├── demo_samples/
│   ├── sample_0_walk.npy                          # Sampel uji asli kelas Walk (matriks 250x90)
│   ├── sample_1_sit.npy                           # Sampel uji asli kelas Sit (matriks 250x90)
│   ├── sample_2_stand.npy                         # Sampel uji asli kelas Stand (matriks 250x90)
│   └── sample_3_fall.npy                          # Sampel uji asli kelas Fall (matriks 250x90)
├── plots/
│   ├── preprocessing_analysis.png                 # Visualisasi Butterworth Filter & Spektrogram STFT (300 DPI)
│   ├── model_comparison_curves.png                # Grafik Training vs Validation Loss & Accuracy 50 Epoch (300 DPI)
│   └── confusion_matrix.png                       # Confusion Matrix jumlah sampel & persentase baris pada Test Set
├── saved_models/
│   ├── baseline_1dcnn.pth                         # Bobot final terpilih Baseline 1D-CNN
│   ├── proposed_cnnlstm.pth                       # Bobot final terpilih Proposed 1D-CNN + Bi-LSTM (Attention)
│   ├── baseline_1dcnn_raw_best.pth                # Checkpoint studi ablasi Baseline (Raw CSI)
│   ├── baseline_1dcnn_filtered_best.pth           # Checkpoint studi ablasi Baseline (Filtered CSI)
│   ├── proposed_cnnbilstm_raw_best.pth            # Checkpoint studi ablasi Proposed (Raw CSI)
│   ├── proposed_cnnbilstm_filtered_best.pth       # Checkpoint studi ablasi Proposed (Filtered CSI)
│   └── norm_params.npy                            # Parameter [mean_train, std_train, flag_filter] dari Training Set
├── PML_Kelompok_5.ipynb                           # Notebook utama Google Colab (Full pipeline, eksperimen, & analisis)
├── preprocessing.py                               # Modul fungsi pengolahan sinyal CSI (Butterworth filter, STFT, dll)
├── models.py                                      # Definisi kelas PyTorch Baseline1DCNN & ProposedCNNBiLSTM
├── train_eval.py                                  # Script CLI untuk verifikasi bobot model & uji inferensi sampel
├── requirements.txt                               # Daftar pustaka Python beserta versinya
└── README.md                                      # Dokumentasi lengkap repositori proyek
```

### Fungsi Setiap File:
1. **`PML_Kelompok_5.ipynb`:** Notebook interaktif Google Colab yang memuat seluruh alur kerja eksperimen dari awal hingga akhir (ekstraksi `UT_HAR.zip`, partisi blok kontigu dengan *guard gap*, *Butterworth filtering*, pelatihan 50 *epoch* dengan GPU CUDA, tabel studi ablasi validasi, evaluasi *single-shot* pada *Test Set*, hingga ekspor grafik dan bobot model).
2. **`preprocessing.py`:** Modul Python mandiri yang berisi fungsi-fungsi pemrosesan sinyal digital:
   * `extract_csi()`: Mengekstrak komponen amplitudo (dan fasa jika input kompleks) dari matriks CSI.
   * `apply_butterworth_lowpass()`: Menerapkan *4th-order Butterworth Low-pass Filter* tanpa pergeseran fasa (`filtfilt`).
   * `sanitize_phase()`: Fungsi pembuka bungkusan fasa (*phase unwrapping*) dan sanitasi linier antar-*subcarrier*.
   * `compute_stft_spectrogram()`: Menghapus *DC offset* dan menghitung spektrogram waktu-frekuensi STFT.
3. **`models.py`:** Modul cetak biru (*blueprint*) arsitektur jaringan saraf tiruan menggunakan `torch.nn.Module`. Memuat kelas `Baseline1DCNN` dan `ProposedCNNBiLSTM` lengkap dengan lapisan *Temporal Attention*. Jika dieksekusi langsung (`python models.py`), file ini akan melakukan *sanity check* dimensi tensor input $(2, 90, 250) \rightarrow$ output $(2, 4)$.
4. **`train_eval.py`:** Script evaluasi dan pengujian inferensi berbasis terminal (*Command Line Interface* / CLI). Memuat bobot model terlatih dari `saved_models/` beserta parameter normalisasi `norm_params.npy`, lalu menjalankan pembersihan *Butterworth filter* dan membandingkan hasil tebakan `Baseline1DCNN` vs `ProposedCNNBiLSTM` pada ke-4 file sampel di dalam folder `demo_samples/`.
5. **`saved_models/`:** Direktori penyimpanan seluruh *state dictionary* bobot PyTorch (`.pth`) dari *checkpoint* validasi terbaik serta parameter normalisasi data latih (`norm_params.npy`).
6. **`demo_samples/`:** Berisi 4 file array NumPy (`.npy`) berdimensi $(250, 90)$ yang diambil dari partisi *Test Set* untuk keempat kelas aktivitas, sehingga pengujian inferensi dapat langsung dijalankan di komputer lokal tanpa perlu mengunduh ulang seluruh dataset mentah.
7. **`plots/`:** Menyimpan hasil ekspor grafik beresolusi tinggi (300 DPI) yang digunakan pada laporan teknis IEEE.

---

## 8. Panduan Instalasi & Cara Menjalankan Kode

### A. Menjalankan Eksperimen Penuh di Google Colab (Direkomendasikan untuk Reproduksi Training)
1. Buka file **`PML_Kelompok_5.ipynb`** menggunakan Google Colab.
2. Unggah file dataset `UT_HAR.zip` ke Google Drive Anda dan sesuaikan variabel `zip_path` pada **Sel 2**.
3. Aktifkan akselerator perangkat keras GPU melalui menu **Runtime > Change runtime type > T4 GPU**.
4. Klik **Runtime > Run all** untuk mengeksekusi seluruh tahapan mulai dari ekstraksi data, pelatihan 50 *epoch*, studi ablasi, hingga pembuatan grafik evaluasi.

### B. Menjalankan Verifikasi Arsitektur & Inferensi di Komputer Lokal (Terminal / CLI)
Pastikan Python 3.9+ telah terinstal di komputer Anda, kemudian ikuti langkah berikut:

1. **Kloning repositori ini dan masuk ke direktori proyek:**
   ```bash
   git clone https://github.com/[username-anda]/wifi-csi-har.git
   cd wifi-csi-har
   ```

2. **Instal seluruh pustaka yang dibutuhkan melalui `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verifikasi dimensi arsitektur model PyTorch (`models.py`):**
   ```bash
   python models.py
   ```
   *Output yang diharapkan:*
   ```text
   Baseline1DCNN Output Shape : torch.Size([2, 4])
   ProposedCNNBiLSTM Out Shape: torch.Size([2, 4])
   ```

4. **Jalankan pengujian inferensi pada sampel uji (`train_eval.py`):**
   ```bash
   python train_eval.py
   ```
   *Output yang diharapkan:*
   ```text
   === Verifikasi Bobot Model & Evaluasi Sampel Uji ===
   File: demo_samples/sample_0_walk.npy  -> Baseline: Walk   | Proposed: Walk  
   File: demo_samples/sample_1_sit.npy   -> Baseline: Sit    | Proposed: Sit   
   File: demo_samples/sample_2_stand.npy -> Baseline: Stand  | Proposed: Stand 
   File: demo_samples/sample_3_fall.npy  -> Baseline: Fall   | Proposed: Fall  
   ```

---

## 9. Referensi Ilmiah Utama
1. **P. F. Moshiri, R. Shahbazian, M. Nabati, and S. A. Ghorashi**, "A CSI-based human activity recognition using deep learning," *Sensors*, vol. 21, no. 21, p. 7225, Oct. 2021, doi: 10.3390/s21217225.
2. **D. Varga**, "Critical analysis of data leakage in WiFi CSI-based human action recognition using CNNs," *Sensors*, vol. 24, no. 10, p. 3159, May 2024, doi: 10.3390/s24103159.
3. **D. Varga**, "Mitigating data leakage in a WiFi CSI benchmark for human action recognition," *Sensors*, vol. 24, no. 24, p. 8201, Dec. 2024, doi: 10.3390/s24248201.
4. **S. Yousefi, H. Narui, S. Dayal, S. Ermon, and S. Valaee**, "A survey on behavior recognition using WiFi channel state information," *IEEE Communications Magazine*, vol. 55, no. 10, pp. 98–104, Oct. 2017.
5. **Z. Chen, L. Zhang, C. Jiang, Z. Cao, and W. Cui**, "WiFi CSI based passive human activity recognition using attention based BLSTM," *IEEE Transactions on Mobile Computing*, vol. 18, no. 11, pp. 2714–2724, Nov. 2019.
