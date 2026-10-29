# Bölüm 2 — Görüntü Sınıflandırma Temelleri (MobileNet V2 + LiteRT)

> **Kaynak:** Marcelo Rovai, *Edge AI Engineering — Hands-On with the Raspberry Pi*, s. 82–108
> **Donanım:** Raspberry Pi 5 (4 GB), Camera Module v2 (IMX219)
> **Yazılım:** Raspberry Pi OS (Debian Trixie), Python 3.13.5, LiteRT (`ai-edge-litert`) 2.3.0, NumPy 2.2.4, Pillow 11.1.0, Picamera2 0.3.38
> **Ön koşul:** [Bölüm 1 — Raspberry Pi 5 kurulumu](https://github.com/ebrarklc/raspberry-pi5-edge-ai-setup)

---

## İçindekiler
1. [Öğrenme hedefleri](#1-öğrenme-hedefleri)
2. [Büyük resim: bir görüntünün yolculuğu](#2-büyük-resim-bir-görüntünün-yolculuğu)
3. [Ortam kurulumu (venv, LiteRT, Jupyter)](#3-ortam-kurulumu)
4. [Model: MobileNet V2 (quantized)](#4-model-mobilenet-v2-quantized)
5. [Adım adım çıkarım](#5-adım-adım-çıkarım)
6. [Deney 1 — Thread sayısı](#6-deney-1--thread-sayısı)
7. [Deney 2 — Modeli bir kez yüklemek](#7-deney-2--modeli-bir-kez-yüklemek)
8. [Kamerayla sınıflandırma ve ISP ile ön işleme](#8-kamerayla-sınıflandırma)
9. [Gözlemler: modelin "hataları"](#9-gözlemler-modelin-hataları)
10. [Terimler sözlüğü](#10-terimler-sözlüğü)
11. [Sorun giderme](#11-sorun-giderme)
12. [Mülakat notları](#12-mülakat-notları)

---

## 1. Öğrenme hedefleri
Bu bölümün sonunda öğrenci:
- Bir sınıflandırma modelinin **ön işleme → çıkarım → son işleme** zincirini kendi koduyla kurabilir,
- Quantized (int8/uint8) bir modelin giriş/çıkış parametrelerini okuyup **dequantize** edebilir,
- **Logit** ile **olasılık** farkını ve **softmax**'ı açıklayabilir,
- Çıkarım süresini **doğru** ölçebilir (warm-up, çoklu ölçüm, adil karşılaştırma),
- Thread sayısının ve ISP'nin performansa etkisini ölçüp yorumlayabilir.

---

## 2. Büyük resim: bir görüntünün yolculuğu

```
① Cat03.jpg          (1600, 1600, 3) uint8, 0–255
        ↓ resize
② Küçültülmüş         (224, 224, 3) uint8
        ↓ expand_dims
③ Model girişi        (1, 224, 224, 3) uint8
        ↓ invoke()   ← ~10 ms (4 thread)
④ Ham çıktı           (1001,) uint8
        ↓ dequantize: (q − 58) × 0.0989
⑤ Logit'ler           (1001,) float32, −3.66 … 11.37
        ↓ softmax
⑥ Olasılıklar         toplam = 1.0
        ↓ argsort + labels.txt
⑦ Sonuç               "tiger cat %39.5"
```

①→③ **ön işleme**, ④ **model**, ⑤→⑦ **son işleme**.
Herhangi bir adım yanlış yapılırsa model **hata vermez**, ama sonuç **anlamsız** olur.

---

## 3. Ortam kurulumu

### 3.1 Sistem paketleri (apt)
```bash
sudo apt install -y python3-pip python3-venv python3-picamera2
```

### 3.2 Sanal ortam
```bash
python3 -m venv ~/tflite_env --system-site-packages
source ~/tflite_env/bin/activate        # satır başında (tflite_env) görünür
```
> 💡 **Neden `--system-site-packages`?** Picamera2, libcamera ve donanıma bağlı olduğu için apt ile **sistem tarafına** kuruludur. Bu bayrak venv'in sistem paketlerini **okumasını** sağlar; yeni kurulan paketler ise sadece venv'e gider.
> Doğrulama: `pip install numpy` → `Requirement already satisfied: numpy in /usr/lib/python3/dist-packages`

### 3.3 Python paketleri (sudo YOK)
```bash
pip install numpy pillow matplotlib opencv-python
pip install ai-edge-litert
pip install jupyter jupyterlab notebook
pip list | grep -E "(numpy|pillow|opencv|picamera)"
```

### 3.4 Çalışma klasörü ve model
```bash
mkdir -p ~/Documents/TFLITE/IMG_CLASS/models
cd ~/Documents/TFLITE/IMG_CLASS/models

wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
tar xzf mobilenet_v2_1.0_224_quant.tgz
wget https://raw.githubusercontent.com/Mjrovai/EdgeML-with-Raspberry-Pi/refs/heads/main/IMG_CLASS/models/labels.txt

# Sadece .tflite ve labels.txt lazım
rm -f mobilenet_v2_1.0_224_quant.ckpt* mobilenet_v2_1.0_224_quant_eval.pbtxt \
      mobilenet_v2_1.0_224_quant_frozen.pb mobilenet_v2_1.0_224_quant.tgz
ls -lh   # mobilenet_v2_1.0_224_quant.tflite → 3.5M
```

### 3.5 Kurulum testi
`~/Documents/TFLITE/IMG_CLASS/setup_test.py`:
```python
from ai_edge_litert.interpreter import Interpreter
import numpy as np
from PIL import Image

print("NumPy:", np.__version__)
print("Pillow:", Image.__version__)

interpreter = Interpreter(model_path="./models/mobilenet_v2_1.0_224_quant.tflite")
interpreter.allocate_tensors()
print("LiteRT Interpreter created successfully!")
```
```
NumPy: 2.2.4
Pillow: 11.1.0
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
LiteRT Interpreter created successfully!
```

### 3.6 Jupyter
```bash
cd ~/Documents/TFLITE/IMG_CLASS
jupyter notebook --ip=192.168.1.42 --no-browser
```
Terminaldeki `http://192.168.1.42:8888/tree?token=...` linkini bilgisayarın tarayıcısında açın. **Terminal açık kalmalıdır.**
Kapatmak için: terminalde `Ctrl+C` → `y`.

---

## 4. Model: MobileNet V2 (quantized)

| Özellik | Değer |
|---|---|
| Mimari | MobileNet V2 (depthwise separable convolution) |
| Eğitim verisi | ImageNet — 1.2 milyon görüntü |
| Sınıf sayısı | 1001 (1000 nesne + 1 arka plan) |
| Parametre | ~3.5 milyon |
| Dosya boyutu | **3.5 MB** (float32 olsaydı ~14 MB) |

**Boyut hesabı:** 3.5 M parametre × **1 byte** (uint8) ≈ 3.5 MB → dosya adındaki **"quant"** = quantized.

### Modelin giriş/çıkış bilgileri
```
GİRİŞ : [  1 224 224   3] uint8 | quantization: (0.0078125, 128)
ÇIKIŞ : [   1 1001]       uint8 | quantization: (0.09889253973960876, 58)
```

**Quantization formülü:** `gerçek = scale × (q − zero_point)`

Giriş için:
| q (piksel) | Gerçek değer |
|---|---|
| 0 | 0.0078125 × (0 − 128) = **−1.0** |
| 128 | **0.0** |
| 255 | 0.0078125 × (255 − 128) = **+0.99** |

➡️ MobileNet eğitimde pikselleri **[−1, +1]** aralığında gördü. Bu normalizasyon quantization parametrelerinin içinde saklı olduğu için ham 0–255 pikselleri **doğrudan** veriyoruz.

> ⚠️ Float32 bir modele ham 0–255 pikseller verilirse hata alınmaz ama sonuçlar saçma olur. Modelin beklediği normalizasyonu **her zaman** kontrol edin.

---

## 5. Adım adım çıkarım

```python
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ai_edge_litert.interpreter import Interpreter

# Modeli yükle
interpreter = Interpreter(model_path="./models/mobilenet_v2_1.0_224_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ön işleme
img = Image.open("Cat03.jpg").convert("RGB")      # RGBA → RGB güvenliği
h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
input_data = np.expand_dims(np.array(img.resize((w, h))), axis=0)
print(input_data.shape, input_data.dtype)         # (1, 224, 224, 3) uint8

# Çıkarım
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])[0]

# Top-5 (ham değerlerle)
labels = [l.strip() for l in open("./models/labels.txt")]
top_k = np.argsort(predictions)[::-1][:5]

# Dequantize → logit
scale, zero_point = output_details[0]['quantization']
logits = (predictions.astype(np.float32) - zero_point) * scale
print(logits.min(), logits.max(), logits.sum())   # -3.66  11.37  536.49 → olasılık DEĞİL

# Sayısal olarak kararlı softmax
e = np.exp(logits - logits.max())
probs = e / e.sum()
for i in top_k:
    print(f"{labels[i]:20s} {probs[i]*100:5.1f}%")
```

**Sonuç (Cat03.jpg):**
| Sınıf | İndeks | Ham (uint8) | Olasılık |
|---|---|---|---|
| tiger cat | 283 | 173 | **39.5%** |
| Egyptian cat | 286 | 169 | 26.6% |
| tabby | 282 | 165 | 17.9% |
| lynx | 288 | 158 | 9.0% |
| carton | 479 | 144 | 2.2% |

İlk dört tahminin hepsi kedigil ✅

---

## 6. Deney 1 — Thread sayısı

```python
for n in [1, 2, 4]:
    itp = Interpreter(model_path=model_path, num_threads=n)
    itp.allocate_tensors()
    itp.set_tensor(itp.get_input_details()[0]['index'], input_data)
    itp.invoke()  # warm-up
    t = []
    for _ in range(20):
        t0 = time.perf_counter(); itp.invoke(); t.append((time.perf_counter() - t0) * 1000)
    print(f"num_threads={n}: {np.mean(t):5.1f} ± {np.std(t):.1f} ms")
```

| Thread | Süre | FPS | Hızlanma | İdeal |
|---|---|---|---|---|
| 1 (varsayılan) | 38.2 ± 2.5 ms | ~26 | 1.00x | 1x |
| 2 | 19.3 ± 0.0 ms | ~52 | 1.98x | 2x |
| 4 | **10.1 ± 0.1 ms** | **~99** | **3.78x** | 4x |

**Yorum:**
- Python'da `num_threads` belirtilmezse LiteRT **tek çekirdek** kullanır. Tek parametreyle **~4 kat** hız kazanılır.
- Konvolüsyon kolay paralelleşen bir işlemdir, bu yüzden hızlanma neredeyse doğrusaldır.
- Tam 4x olmamasının sebepleri: **Amdahl Yasası** (paralel olmayan kısım), **ortak bellek bant genişliği**, **thread senkronizasyonu**.
- Amdahl: 3.78x hızlanma → işin yaklaşık **%98'i paralel**.
- 4 thread'de sapma ±2.5 ms'den ±0.1 ms'ye düştü: ölçüm çok daha kararlı.

---

## 7. Deney 2 — Modeli bir kez yüklemek

Dokümandaki `image_classification()` fonksiyonu **her çağrıda** modeli yükler ve belleği yeniden ayırır. Doğru tasarım: **pahalı işi bir kez yap.**

```python
class ImageClassifier:
    def __init__(self, model_path, labels_path, num_threads=4):
        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.h, self.w = self.inp['shape'][1], self.inp['shape'][2]
        self.scale, self.zero_point = self.out['quantization']
        with open(labels_path) as f:
            self.labels = [line.strip() for line in f]

    def classify(self, img, top_k=5):
        t0 = time.perf_counter()
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        x = np.expand_dims(np.array(img.convert("RGB").resize((self.w, self.h))), axis=0)
        t1 = time.perf_counter()
        self.interpreter.set_tensor(self.inp['index'], x)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.out['index'])[0]
        t2 = time.perf_counter()
        logits = (raw.astype(np.float32) - self.zero_point) * self.scale
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        top = np.argsort(probs)[::-1][:top_k]
        t3 = time.perf_counter()
        results = [(self.labels[i], float(probs[i])) for i in top]
        times = {"pre": (t1-t0)*1000, "infer": (t2-t1)*1000, "post": (t3-t2)*1000}
        return results, times
```

### Süre dağılımı (1600×1600 JPEG dosyasından)
| Aşama | Süre |
|---|---|
| Ön işleme (JPEG decode + resize) | **~50 ms** 🐢 |
| Çıkarım | ~13 ms |
| Son işleme | ~0.2 ms |

➡️ Bu senaryoda **darboğaz model değil, ön işleme**: 2.56 milyon pikselin decode edilip küçültülmesi, 50 bin piksellik modelden daha pahalı.

### Adil karşılaştırma dersi
| Test | Her seferinde yükle | Bir kez yükle | Kazanç |
|---|---|---|---|
| ❌ Adil değil (biri ön işleme dahil, diğeri değil) | 35.5 ms | 50.2 ms | 0.7x |
| ✅ Adil (ikisi de aynı küçültülmüş görüntü) | 32.2 ms | **10.7 ms** | **3.0x** |

➡️ Model yükleme + `allocate_tensors` her seferinde **~22 ms** tutuyor (çıkarımın 2 katı).
➡️ Karşılaştırılan iki yöntem arasında **sadece tek bir şey** farklı olmalı; yoksa ölçülen şey, sanılan şey değildir.

---

## 8. Kamerayla sınıflandırma

**Fikir:** Küçültmeyi CPU'ya değil, kameranın **ISP**'sine (donanım görüntü işlemcisi) yaptır.

```python
from picamera2 import Picamera2

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "BGR888"},   # ISP doğrudan küçük görüntü versin
    sensor={"output_size": (1640, 1232)}              # tam görüş alanı modu
)
picam2.configure(config)
picam2.start()
time.sleep(2)  # otomatik pozlama (AE) otursun

clf = ImageClassifier("./models/mobilenet_v2_1.0_224_quant.tflite", "./models/labels.txt")

frame = picam2.capture_array()          # (240, 320, 3) RGB numpy dizisi
results, times = clf.classify(frame)

plt.imshow(frame); plt.title(f"{results[0][0]} ({results[0][1]*100:.0f}%)"); plt.axis('off'); plt.show()

# İş bitince
picam2.stop()
picam2.close()
```

Log doğrulaması:
```
configuring streams: (0) 320x240-BGR888/sRGB (1) 1640x1232-BGGR_PISP_COMP1/RAW
Selected sensor format: 1640x1232-SBGGR10_1X10/RAW
```

> ⚠️ **Picamera2 renk sırası tuzağı:** İsimler libcamera'dan geldiği için terstir.
> `"RGB888"` → numpy'da **[B, G, R]**, `"BGR888"` → numpy'da **[R, G, B]**.
> Modele RGB vermek için **`BGR888`** seçilir. Yanlış seçimde hata alınmaz ama renkler (ve tahminler) bozulur.

> ⚠️ Picamera2'nin varsayılan önizleme ayarı **640×480 kırpılmış** sensör modunu seçer (görüntü "yakın" çıkar). Tam görüş alanı için `sensor={"output_size": (1640, 1232)}` kullanın.

### Sonuçlar
| Ölçüm | Dosyadan (1600×1600 JPEG) | Kameradan (ISP, 320×240) |
|---|---|---|
| Ön işleme | ~50 ms | **2.5–2.7 ms** (≈20x daha hızlı) |
| Çıkarım | ~10–13 ms | 15.5–17.9 ms* |

\* Kamera açıkken ISP sürücüsü, kamera thread'leri ve çizim aynı çekirdekleri paylaşır; gerçek sistem ölçümü izole benchmark'tan biraz kötüdür.

---

## 9. Gözlemler: modelin "hataları"

| Görüntü | Tahmin | Ders |
|---|---|---|
| Bulanık yakın çekim | switch **%9** | Camera v2'nin odağı sabittir; çok yakın nesne bulanık çıkar. Düşük güven = model "emin değilim" diyor. |
| Raspberry Pi'nin kendisi | modem %25, monitor %18 | ImageNet'te "Raspberry Pi" sınıfı **yok**. Model bildiği 1000 sınıftan en benzerini seçer (**closed-set** problemi). |
| Masa sahnesi (klavye, telefon, kâğıt...) | desk %38, notebook %19, laptop %16 | Sınıflandırma bütün resme **tek etiket** verir; birden çok nesnede olasılık bölünür. |

➡️ **Sonraki bölümlerin motivasyonu:**
- **Bölüm 3:** ImageNet'te olmayan **kendi sınıflarımızı** öğretmek (custom classification).
- **Bölüm 4:** Sahnedeki her nesneyi **konumuyla birlikte** ayrı ayrı bulmak (object detection).

---

## 10. Terimler sözlüğü

### Model ve öğrenme
| Terim | Açıklama |
|---|---|
| **Model** | Girdiyi (görüntü) çıktıya (sınıf) dönüştüren, milyonlarca sayıdan oluşan matematiksel fonksiyon. |
| **Parametre / ağırlık** | Modelin eğitimde öğrendiği ve davranışını belirleyen sayılar; MobileNet V2'de ~3.5 milyon. |
| **Eğitim (training)** | Modele etiketli örnekler gösterip ağırlıklarını hatayı azaltacak şekilde ayarlama süreci. |
| **Çıkarım (inference)** | Eğitilmiş modeli yeni bir girdi üzerinde çalıştırıp tahmin almak. |
| **Pre-trained model** | Başkasının büyük bir veri setiyle eğitip paylaştığı hazır model. |
| **ImageNet** | 1.2 milyon fotoğraf ve 1000 sınıftan oluşan görüntü sınıflandırma veri seti. |
| **Sınıf / etiket** | Modelin seçebileceği kategorilerden biri; `labels.txt` indeksleri isimlere çevirir. |
| **Sınıflandırma** | Görüntünün tamamına tek bir etiket verme; nesnenin konumunu söylemez. |

### Mimari
| Terim | Açıklama |
|---|---|
| **Mimari** | Modelin katmanlarının nasıl dizildiğini tanımlayan tasarım planı. |
| **Konvolüsyon** | Küçük bir filtreyi görüntü üzerinde kaydırıp kenar, doku, şekil gibi desenleri yakalayan işlem. |
| **CNN** | Konvolüsyon katmanları üzerine kurulu sinir ağı. |
| **Depthwise separable convolution** | Konvolüsyonu "kanal başına filtre" + "1×1 ile kanalları karıştır" diye iki ucuz adıma bölen, ~8–9 kat daha az hesap yapan teknik. |
| **MobileNet V2** | Depthwise separable convolution kullanan, mobil ve edge için tasarlanmış hafif CNN. |

### Veri ve tensörler
| Terim | Açıklama |
|---|---|
| **Tensör** | Çok boyutlu sayı dizisi. |
| **Shape** | Tensörün her boyuttaki uzunluğu, örn. (1, 224, 224, 3). |
| **H, W, C** | Yükseklik (satır), genişlik (sütun), kanal (RGB için 3). |
| **Kanal** | Pikselin bir renk bileşeni (R, G veya B). |
| **Batch boyutu** | Aynı anda kaç görüntü işlendiğini belirten ilk boyut. |
| **dtype** | Sayıların saklanma tipi: `uint8` (0–255, 1 byte), `float32` (4 byte). |
| **Ön işleme** | Görüntüyü modelin beklediği şekle ve tipe getirme. |
| **Normalizasyon** | Pikselleri modelin eğitimde gördüğü aralığa ölçekleme. |

### Quantization
| Terim | Açıklama |
|---|---|
| **Quantization** | Ağırlık ve aktivasyonları 8-bit tam sayıyla saklayıp modeli ~4 kat küçültme ve hızlandırma. |
| **Scale** | Bir tam sayı adımının gerçek dünyadaki karşılığı. |
| **Zero point** | Gerçek sıfıra karşılık gelen tam sayı. |
| **Dequantize** | uint8 çıktıyı `scale × (q − zero_point)` ile gerçek değere çevirme. |
| **PTQ** | Post-Training Quantization: eğitimden sonra kalibrasyon verisiyle niceleme. |
| **QAT** | Quantization-Aware Training: eğitim sırasında nicelemeyi taklit ederek doğruluk kaybını azaltma. |

### Çıktı ve son işleme
| Terim | Açıklama |
|---|---|
| **Logit** | Her sınıf için ham, ölçeklenmemiş puan; negatif olabilir, toplamı 1 değildir. |
| **Softmax** | Logit'leri pozitif ve toplamı 1 olan olasılıklara çeviren fonksiyon: e^x / Σe^x. |
| **Overflow** | Sayının veri tipi sınırını aşması (e^1000 → inf). |
| **Sayısal kararlılık** | Sonucu değiştirmeden taşmayı önleme (softmax'ta max'ı çıkarmak). |
| **argsort** | Değerlerin sıralı **indekslerini** döndüren numpy fonksiyonu. |
| **Top-k** | En yüksek olasılıklı k sınıf. |

### Çalışma ortamı
| Terim | Açıklama |
|---|---|
| **LiteRT** | Google'ın edge cihazlar için model çalıştırma motoru (TensorFlow Lite'ın yeni adı). |
| **.tflite** | Model mimarisi ve ağırlıklarını tek dosyada tutan kompakt format (FlatBuffer). |
| **Interpreter** | `.tflite` dosyasını okuyup modeli çalıştıran LiteRT nesnesi. |
| **allocate_tensors()** | Tüm tensörler için belleği önceden bir kez ayırma. |
| **invoke()** | Modeli bir kez çalıştırma. |
| **Delegate** | İşlemleri daha hızlı bir arka uca (CPU kütüphanesi, GPU, NPU) devretme mekanizması. |
| **XNNPACK** | ARM/x86 CPU'lar için optimize edilmiş sinir ağı işlem kütüphanesi. |
| **NPU / DSP** | Sinir ağı / sinyal işleme için özelleşmiş düşük güçlü işlemci birimleri. |
| **ISP** | Kameranın ham sensör verisini işleyen, ölçekleme ve renk dönüşümünü donanımda yapan görüntü işlemcisi. |

### Performans
| Terim | Açıklama |
|---|---|
| **Gecikme (latency)** | Tek bir girdinin işlenme süresi. |
| **FPS / throughput** | Saniyede işlenen görüntü sayısı (≈ 1000 / gecikme ms). |
| **Warm-up** | Ölçümden önce tek seferlik maliyetleri ayıklamak için yapılan ilk çalıştırma. |
| **Çekirdek (core)** | İşlemcinin bağımsız birimi; Pi 5'te 4 × Cortex-A76. |
| **Thread** | Paralel çalışabilen alt görev. |
| **Amdahl Yasası** | Paralel olmayan kısmın toplam hızlanmayı sınırlaması: `1 / ((1−p) + p/n)`. |

---

## 11. Sorun giderme

| Sorun | Sebep / Çözüm |
|---|---|
| `pip install` sonrası kırmızı `ERROR: ... types-seaborn requires pandas-stubs` | Sistemdeki tip ipucu paketiyle ilgili, **zararsız**. Son satır `Successfully installed` ise sorun yok. |
| `Can't uninstall 'typing_extensions'... outside environment` | pip sistem paketini silmeye kalkışmadı, venv'e yeni sürümü kurdu. **Zararsız.** |
| Jupyter'da kırmızı kutular (libcamera logları) | `INFO`/`WARN` logları stderr'e yazıldığı için kırmızı görünür; hata değildir. |
| `Camera in use` / kamera açılmıyor | Başka bir süreç (ör. `rpicam-hello`) kamerayı kullanıyor. Kapatın veya `picam2.close()` çağırın. |
| İlk kare karanlık | Otomatik pozlama oturmadı; `start()` sonrası `time.sleep(2)`. |
| Görüntü "zoom yapılmış" gibi | Picamera2 varsayılan olarak 640×480 kırpılmış modu seçer → `sensor={"output_size": (1640, 1232)}`. |
| Renkler ters (portakal mavi) | `"RGB888"` yerine `"BGR888"` kullanın. |
| Çıkarım beklenenden yavaş | `Interpreter(..., num_threads=4)` verilmiş mi? Varsayılan tek thread'dir. |
| Jupyter yeniden açılınca `NameError` | Kernel yeniden başladı; **Run → Run All Cells**. |
| Yeni terminalde `ModuleNotFoundError` | venv aktif değil: `source ~/tflite_env/bin/activate`. |

---

## 12. Mülakat notları

**"Quantization neden yapılır, bedeli ne?"**
Model ~4 kat küçülür, bellek bant genişliği azalır, int8 aritmetiği hızlıdır ve NPU/DSP'ler genelde yalnızca int8 çalıştırır. Bedeli genelde %1–2 doğruluk kaybıdır. PTQ kolaydır; QAT doğruluğu daha iyi korur.

**"Quantized bir modelin girişine neden ham 0–255 piksel veriyoruz?"**
Normalizasyon (ör. [−1, +1]) giriş tensörünün scale/zero_point değerlerinde saklıdır: `0.0078125 × (q − 128)`.

**"Logit ve olasılık farkı nedir?"**
Logit, ham ölçeklenmemiş puandır (negatif olabilir, toplamı 1 değildir). Softmax ile olasılığa çevrilir.

**"Softmax'ı numpy ile yaz."**
```python
e = np.exp(x - np.max(x)); p = e / e.sum()
```
`max`'ı çıkarmak sonucu değiştirmez, overflow'u önler (sayısal kararlılık).

**"Delegate nedir?"**
Modelin işlemlerini daha uygun bir donanıma veya kütüphaneye devretme mekanizması (XNNPACK, GPU, NNAPI, Edge TPU...).

**"4 çekirdekte neden tam 4 kat hızlanmıyoruz?"**
Amdahl Yasası (paralel olmayan kısım), ortak bellek bant genişliği ve thread senkronizasyonu.

**"Gerçek zamanlı bir sistemde neyi bir kez, neyi her karede yaparsın?"**
Model yükleme, bellek ayırma, kamera açma **bir kez**; sadece ön işleme → çıkarım → son işleme **her karede**.

**"Edge'de ön işlemeyi nasıl hızlandırırsın?"**
İşi donanıma yaptır (ISP'den hedef çözünürlükte görüntü iste), JPEG encode/decode'u atla, ham diziyle çalış.

**"Adil benchmark nasıl yapılır?"**
Warm-up, çoklu ölçüm, ortalama ± std ve karşılaştırılan yöntemler arasında **tek bir fark**.

**"Closed-set problemi nedir?"**
Sınıflandırıcı görmediği bir şeyle karşılaşınca "bilmiyorum" demez, bildiği sınıflardan birini seçer. Çözümler: güven eşiği, "diğer/arka plan" sınıfı, OOD tespiti.
