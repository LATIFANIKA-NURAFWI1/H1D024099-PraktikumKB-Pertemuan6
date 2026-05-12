
# Praktikum Kecerdasan Buatan Pertemuan 6 materi Jaringan Syaraf Tiruan (JST) – Perceptron & Backpropagation
  
**Nama**: LATIFANIKA NURAFWI 
**NIM** HID024099

- **Perceptron** untuk menyelesaikan masalah logika **OR**
- **Backpropagation** untuk menyelesaikan masalah logika **XOR**

### 1. Perceptron
Perceptron adalah model jaringan syaraf **single layer** yang menggunakan fungsi aktivasi **bipolar** (keluaran `1` atau `-1`).  
Model ini belajar dengan aturan **Delta Rule**: bobot diperbarui jika prediksi tidak sesuai target.  
Cocok untuk masalah yang **linearly separable** seperti gerbang OR.

### 2. Backpropagation
Backpropagation adalah jaringan **multilayer** (ada *hidden layer*) yang memanfaatkan propagasi maju untuk menghitung keluaran dan propagasi mundur untuk memperbaiki bobot.  
Fungsi aktivasi yang digunakan adalah **tanh (sigmoid bipolar)**.  
Algoritma ini mampu menyelesaikan masalah **non-linear** seperti XOR.

---

## Struktur Folder

```
H1D024099-PraktikumKB-Pertemuan6/
│
├── Perceptron.py                # Kelas Perceptron (model, prediksi, plotting)
├── Perceptron_or.py             # Skrip untuk menjalankan Perceptron pada data OR
├── Backpropagation.py           # Kelas Backpropagation (multilayer, training)
├── Backpropagation_xor.py       # Skrip untuk menjalankan Backpropagation pada data XOR
├── HasilPerceptron.txt          # Log pelatihan Perceptron
├── HasilBackpropagation.txt     # Log pelatihan Backpropagation (epoch, error, dll.)
├── Perceptron_1.png             # Decision boundary epoch 1
├── Perceptron_2.png             # Decision boundary epoch 2
├── Perceptron_3.png             # Decision boundary epoch 3
├── Backpropagation_1.png        # Grafik penurunan error (contoh 1)
├── Backpropagation_2.png        # Grafik penurunan error (contoh 2)
├── Backpropagation_3.png        # Grafik penurunan error (contoh 3)
└── README.md                    # Penjelasan program
```

---

## Penjelasan Kode 

### `Perceptron.py`
Kelas **Perceptron** berisi semua logika untuk pelatihan dan visualisasi.

| Metode | Fungsi Singkat |
|--------|----------------|
| `__init__(alpha, epoch)` | Menyimpan learning rate dan jumlah maksimum epoch. |
| `weighted_sum(X)` | Menghitung `y_in = b + Σ(x_i * w_i)`. |
| `predict(X)` | Menerapkan fungsi aktivasi bipolar: jika `weighted_sum ≥ 0` → `1`, selain itu `-1`. |
| `plot_decision_boundary(X, t, epoch)` | Menggambar titik data dan garis pemisah sesuai bobot saat ini. |
| `fit(X, t)` | Melatih model menggunakan aturan Delta Rule: <br> - Bobot awal = 0 <br> - Untuk setiap epoch, hitung prediksi, error, dan perbarui bobot jika error ≠ 0 <br> - Hentikan jika **SSE = 0** atau epoch maksimum tercapai <br> - Tulis log ke `HasilPerceptron.txt` dan tampilkan plot setiap epoch. |

```python
# Update bobot dengan Delta Rule
update = self.alpha * error[-1]
self.w_[1:] += update * xi
self.w_[0] += update
```

### `Perceptron_or.py`
```python
X = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])   # input OR
t = np.array([[1], [1], [1], [-1]])               # target bipolar
model = p.Perceptron(alpha=0.1, epoch=10)
model.fit(X, t)
```
Panggil `fit()` semua proses berjalan otomatis.

### `Backpropagation.py`
Kelas **Backpropagation** (multilayer) dengan arsitektur:
- Input layer: 2 neuron
- Hidden layer: 2 neuron (fungsi aktivasi **tanh**)
- Output layer: 1 neuron (fungsi aktivasi **tanh**)

| Metode | Fungsi Singkat |
|--------|----------------|
| `__init__(alpha, epoch, target_error)` | Inisialisasi parameter, bobot, dan bias (random) untuk hidden & output layer. |
| `bi_sigmoid(x)` | Fungsi aktivasi tanh. |
| `deriv_bi_sigmoid(x)` | Turunan tanh (`1 - x²`), asumsi `x` sudah diaktivasi. |
| `plot_error(x, epoch)` | Menggambar grafik penurunan Sum Square Error (SSE) per epoch. |
| `fit(X, t)` | Melakukan **Forward Propagation** (hitung `h_in`, `h`, `y_in`, `y`) lalu **Backward Propagation** (hitung error, delta output, delta hidden, perbarui bobot). <br> Iterasi berhenti jika **SSE < target_error** atau epoch maksimum tercapai. <br> Semua detail ditulis ke `HasilBackpropagation.txt`. |

**Logika Backpropagation (singkat):**
1. Forward: `input → hidden (tanh) → output (tanh)`
2. Hitung error = `target - output`
3. Backward: hitung delta output → delta hidden → update bobot hidden & output
4. Ulangi untuk setiap data, lalu rata-rata error (SSE) per epoch.

### `Backpropagation_xor.py`
Mirip dengan Perceptron, hanya datanya XOR bipolar:
```python
X = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
t = np.array([[-1], [1], [1], [-1]])   # target XOR bipolar
model = b.Backpropagation(alpha=0.3, epoch=1000, target_error=0.001)
model.fit(X, t)
```

---

## Cara Menjalankan Program

### Persyaratan
- Python 3.x
- Library: `numpy`, `matplotlib`

Install library jika belum ada:
```bash
pip install numpy matplotlib
```

### Jalankan Perceptron (OR)
```bash
python Perceptron_or.py
```
Akan muncul plot garis pemisah setiap epoch, dan file `HasilPerceptron.txt` akan dibuat.

### Jalankan Backpropagation (XOR)
```bash
python Backpropagation_xor.py
```
Akan muncul grafik penurunan error setiap kali pelatihan selesai, dan file `HasilBackpropagation.txt` akan dibuat.

---

## Output

### 1. Output Perceptron
- **`HasilPerceptron.txt`** mencatat:
  - Bobot awal, bias = 0
  - Setiap epoch: input, target, prediksi, error, bobot baru, SSE
  - Pelatihan berhenti di epoch ke-3 karena **SSE = 0** (berhasil memisahkan semua data).
- **Plot decision boundary** (`Perceptron_1.png`, `_2.png`, `_3.png`):
  - Titik merah/biru mewakili target `1`/`-1`
  - Garis biru adalah pemisah yang diperbarui setiap epoch sampai memisahkan kedua kelas dengan sempurna.

### 2. Output Backpropagation
- **`HasilBackpropagation.txt`** sangat detail:
  - Bobot awal (random), learning rate, max epoch
  - Untuk setiap epoch: data ke-1 s.d. 4, forward propagation (h_in, h, y_in, y), backward propagation (error, delta output, delta hidden), bobot baru
  - SSE rata-rata per epoch
  - Pelatihan berhenti di epoch ke-477 karena **SSE < 0.001**.
- **Grafik penurunan error** (`Backpropagation_1.png`, dll.): sumbu Y adalah SSE, sumbu X adalah epoch. Terlihat error menurun tajam di awal lalu melandai hingga mencapai target.

---

## Kesimpulan
- **Perceptron** mampu memisahkan data OR **dalam 3 epoch** karena data bersifat *linearly separable*.
- **Backpropagation** dengan arsitektur 2-2-1 berhasil menyelesaikan XOR **dalam 477 epoch** dengan target error 0.001 dan learning rate 0.3.
- Kedua algoritma menunjukkan bahwa pemilihan model dan arsitektur sangat penting sesuai dengan kompleksitas masalah.