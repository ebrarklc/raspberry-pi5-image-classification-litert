# 🧠 Raspberry Pi 5 — Görüntü Sınıflandırma Temelleri (MobileNet V2 + LiteRT)

**Önceden eğitilmiş, quantized bir MobileNet V2 modelini Raspberry Pi 5'te adım adım çalıştırma: tensörlerin içine bakmak, dequantize, softmax, performans ölçümü ve canlı kamerayla sınıflandırma.**
Edge AI serisinin 2. bölümü.

![Raspberry Pi 5](https://img.shields.io/badge/Raspberry%20Pi-5%20(4GB)-C51A4A?logo=raspberrypi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![LiteRT](https://img.shields.io/badge/LiteRT-2.3.0-FF6F00?logo=tensorflow&logoColor=white)
![Model](https://img.shields.io/badge/Model-MobileNet%20V2%20uint8-4285F4)
![Camera](https://img.shields.io/badge/Camera-IMX219%20v2-2E8B57)
![Dil](https://img.shields.io/badge/Dil-Türkçe-E30A17)

---

## 🎯 Bu repoda neler var?

Bir sınıflandırma modelini "kara kutu" olarak çağırmak yerine, **görüntünün modele girişinden olasılığa dönüşmesine kadar her adımı** kendi kodumuzla kurup ölçtük.

📄 **Tam ders notu:** [02-image-classification-temelleri.md](02-image-classification-temelleri.md)
📓 **Jupyter defteri:** [10_Image_Classification.ipynb](10_Image_Classification.ipynb)

```
📸 Görüntü → 🔧 Ön işleme → 🧠 MobileNet V2 (LiteRT) → 📊 Dequantize + Softmax → 🏷️ "tiger cat %39.5"
```

| # | Konu | Öğrenilenler |
|---|---|---|
| 1 | Ortam | `venv --system-site-packages`, LiteRT, Jupyter |
| 2 | Model | MobileNet V2, ImageNet, quantization (uint8, scale, zero_point) |
| 3 | Çıkarım | Ön işleme, `invoke()`, logit → dequantize → softmax → top-5 |
| 4 | Performans | Thread ölçeklemesi, warm-up, adil benchmark, Amdahl Yasası |
| 5 | Kamera | Picamera2 + ISP ile donanımda ön işleme, renk sırası tuzağı |

---

## 📊 Öne çıkan sonuçlar

### 🐱 Cat03.jpg sınıflandırması
| Sınıf | Olasılık |
|---|---|
| tiger cat | **39.5%** |
| Egyptian cat | 26.6% |
| tabby | 17.9% |
| lynx | 9.0% |
| carton | 2.2% |

### 🧵 Thread sayısı ve çıkarım süresi (Pi 5, Cortex-A76 × 4)
| Thread | Süre | FPS | Hızlanma |
|---|---|---|---|
| 1 *(varsayılan!)* | 38.2 ± 2.5 ms | ~26 | 1.00x |
| 2 | 19.3 ± 0.0 ms | ~52 | 1.98x |
| 4 | **10.1 ± 0.1 ms** | **~99** | **3.78x** |

➡️ `num_threads=4` tek parametresiyle **~4 kat** hız. Amdahl'a göre işin **~%98'i paralel**.

### ⚡ Modeli bir kez yüklemek
| Yöntem | Süre / görüntü |
|---|---|
| Her çağrıda yükle + `allocate_tensors` | 32.2 ms |
| Bir kez yükle, sonra sadece `invoke` | **10.7 ms** (3.0x) |

### 📸 Ön işleme: CPU yerine ISP
| Kaynak | Ön işleme süresi |
|---|---|
| 1600×1600 JPEG (decode + resize, CPU) | ~50 ms |
| Kamera ISP'si doğrudan 320×240 veriyor | **~2.5 ms** (≈20x) |

➡️ Dosyadan çalışırken darboğaz **model değil, ön işlemeydi**. Ölçmeden bilinmez!

---

## 🔑 Önemli bulgular

1. **Quantized modelin normalizasyonu parametrelerin içinde:** giriş `scale=1/128, zero_point=128` → ham 0–255 piksel otomatik olarak [−1, +1] aralığına eşleniyor.
2. **Çıktı olasılık değil, logit:** dequantize sonrası min −3.66, toplam 536 → softmax şart.
3. **LiteRT Python'da varsayılan olarak tek thread kullanıyor.**
4. **Picamera2 renk sırası terstir:** `"BGR888"` → numpy'da `[R, G, B]`.
5. **Picamera2 varsayılan olarak kırpılmış 640×480 sensör modunu seçer**; tam görüş alanı için `sensor={"output_size": (1640, 1232)}`.
6. **Closed-set problemi:** Raspberry Pi'nin kendisini gösterince model "modem" dedi, çünkü ImageNet'te "Raspberry Pi" sınıfı yok. → *Bölüm 3'ün motivasyonu.*

---

## ⚡ Hızlı başlangıç

```bash
# Ortam
sudo apt install -y python3-pip python3-venv python3-picamera2
python3 -m venv ~/tflite_env --system-site-packages
source ~/tflite_env/bin/activate
pip install numpy pillow matplotlib opencv-python ai-edge-litert jupyter

# Model
mkdir -p ~/Documents/TFLITE/IMG_CLASS/models && cd ~/Documents/TFLITE/IMG_CLASS/models
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
tar xzf mobilenet_v2_1.0_224_quant.tgz
wget https://raw.githubusercontent.com/Mjrovai/EdgeML-with-Raspberry-Pi/refs/heads/main/IMG_CLASS/models/labels.txt

# Jupyter
cd ~/Documents/TFLITE/IMG_CLASS
jupyter notebook --ip=<PI_IP_ADRESI> --no-browser
```

### Çekirdek kod
```python
from ai_edge_litert.interpreter import Interpreter
import numpy as np

interpreter = Interpreter(model_path="models/mobilenet_v2_1.0_224_quant.tflite", num_threads=4)
interpreter.allocate_tensors()
inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]

x = np.expand_dims(np.array(img.convert("RGB").resize((224, 224))), axis=0)   # (1,224,224,3) uint8
interpreter.set_tensor(inp['index'], x)
interpreter.invoke()

scale, zp = out['quantization']
logits = (interpreter.get_tensor(out['index'])[0].astype(np.float32) - zp) * scale
probs = np.exp(logits - logits.max()); probs /= probs.sum()               # kararlı softmax
```

Sınıf tabanlı (`ImageClassifier`) sürüm ve kamera entegrasyonu için 👉 [tam ders notu](02-image-classification-temelleri.md)

---

## 📝 Mülakat notları (özet)

- **Quantization neden?** → ~4x küçük, hızlı int8 aritmetiği, NPU uyumu; bedeli ~%1–2 doğruluk.
- **Logit ve olasılık farkı?** → Logit ham puandır; softmax ile olasılığa çevrilir.
- **Kararlı softmax?** → `exp(x − max(x))`; sonucu değiştirmez, overflow'u önler.
- **4 çekirdekte neden tam 4x değil?** → Amdahl, bellek bant genişliği, senkronizasyon.
- **Gerçek zamanlı sistemde ne bir kez yapılır?** → Model yükleme, bellek ayırma, kamera açma.
- **Adil benchmark?** → Warm-up + çoklu ölçüm + ortalama ± std + tek değişken.

Tüm terimler sözlüğü ve mülakat notları ders notunun sonunda.

---

## 🗺️ Edge AI serisi

| # | Bölüm | Repo |
|---|---|---|
| 1 | Raspberry Pi 5 kurulumu | [raspberry-pi5-edge-ai-setup](https://github.com/ebrarklc/raspberry-pi5-edge-ai-setup) |
| 2 | Görüntü sınıflandırma temelleri | 📍 *bu repo* |
| 3 | Kendi görüntü sınıflandırıcın (Edge Impulse) | 🔜 |
| 4 | Nesne tespiti (SSD MobileNet, EfficientDet) | 🔜 |

---

## 📚 Kaynak

Marcelo Rovai — *Edge AI Engineering: Hands-On with the Raspberry Pi* (UNIFEI, 2026), s. 82–108.
Model: [MobileNet V2 1.0 224 quant](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz) (TensorFlow).

## 👤 Hazırlayan

**Ebrar Kılıç** — Edge AI & Embedded Systems · [github.com/ebrarklc](https://github.com/ebrarklc)
