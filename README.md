# 🕳️ Pothole Segmentation Model

**DSConv + SimAM + GELU Enhanced YOLOv11 for Pothole Detection**

Bu proje, çukur (pothole) tespiti ve segmentasyonu için geliştirilmiş özelleştirilmiş bir YOLOv11 modelidir.

## 🌟 Özellikler

### Dynamic Snake Convolution (DSConv)
- Standart konvolüsyonların aksine, kernel yapısını hedef nesnenin şekline dinamik olarak hizalar
- Kıvrımlı ve düzensiz kenarlı çukurların segmentasyonunda daha hassas sonuçlar

### Simple Attention Module (SimAM)
- Parametre eklemeden çalışan dikkat mekanizması
- Çukur vs yama/gölge ayrımını iyileştirir
- Overfitting riskini azaltır

### GELU Aktivasyon Fonksiyonu
- SiLU yerine GELU kullanılarak daha stabil öğrenme
- Karmaşık desenlerde daha iyi performans

## 📁 Proje Yapısı

```
Yolo11Model/
├── ultralytics_custom/          # Custom modüller
│   ├── __init__.py
│   ├── model_builder.py         # Model oluşturucu
│   ├── nn/
│   │   ├── __init__.py
│   │   └── modules/
│   │       ├── __init__.py
│   │       ├── dsconv.py        # Dynamic Snake Convolution
│   │       ├── simam.py         # Simple Attention Module
│   │       ├── conv.py          # GELU Convolution
│   │       └── c3k2_dsconv.py   # C3k2 DSConv blokları
│   └── cfg/
│       └── models/
│           ├── pothole_seg.yaml
│           └── pothole_seg_custom.yaml
├── data/
│   └── pothole.yaml             # Veri seti konfigürasyonu
├── notebooks/
│   └── train_colab.ipynb        # Colab eğitim notebook'u
├── train.py                     # Eğitim scripti
├── requirements.txt
└── README.md
```

## 🚀 Kurulum

### Yerel Ortam

```bash
# Repository'yi klonlayın
git clone https://github.com/YOUR_USERNAME/Yolo11Model.git
cd Yolo11Model

# Bağımlılıkları kurun
pip install -r requirements.txt
```

### Google Colab

1. Repository'yi GitHub'a yükleyin
2. `notebooks/train_colab.ipynb` dosyasını Colab'da açın
3. Adımları takip edin

## 📊 Veri Seti Hazırlığı

Veri setinizi aşağıdaki yapıda hazırlayın:

```
data/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   └── val/
│       ├── img001.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt
    │   └── ...
    └── val/
        ├── img001.txt
        └── ...
```

### Label Formatı (YOLO Segment)

Her satır: `class_id x1 y1 x2 y2 x3 y3 ...` (normalize edilmiş polygon koordinatları)

Örnek:
```
0 0.5 0.3 0.6 0.35 0.65 0.4 0.6 0.45 0.5 0.4
```

## 🏋️ Eğitim

### Yerel Eğitim

```bash
python train.py --data data/pothole.yaml --epochs 100 --batch 16 --imgsz 640
```

### Parametreler

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--data` | - | Veri seti YAML dosyası |
| `--epochs` | 100 | Epoch sayısı |
| `--batch` | 16 | Batch size |
| `--imgsz` | 640 | Görüntü boyutu |
| `--lr0` | 0.01 | Başlangıç learning rate |
| `--device` | auto | Cihaz (cuda:0, cpu) |
| `--modify-act` | True | Aktivasyonları GELU'ya çevir |

### Colab Eğitimi

`notebooks/train_colab.ipynb` dosyasını kullanın.

## 🔬 Custom Modüller

### DSConv Kullanımı

```python
from ultralytics_custom.nn.modules import DSConv, DySnakeConv

# Tek yönlü DSConv
dsconv = DSConv(in_channels=64, out_channels=128, kernel_size=9, morph=0)

# İki yönlü DySnakeConv
dysnake = DySnakeConv(in_channels=64, out_channels=128)
```

### SimAM Kullanımı

```python
from ultralytics_custom.nn.modules import SimAM

# Parametre-free attention
simam = SimAM(e_lambda=1e-4)
output = simam(feature_map)
```

### GELU Conv Kullanımı

```python
from ultralytics_custom.nn.modules import ConvGELU

# GELU aktivasyonlu konvolüsyon
conv = ConvGELU(c1=64, c2=128, k=3, s=1)
```

## 📈 Sonuçlar

Eğitim tamamlandıktan sonra sonuçlar `runs/pothole_seg/train/` klasöründe bulunur:

- `weights/best.pt` - En iyi model ağırlıkları
- `weights/last.pt` - Son epoch ağırlıkları
- `results.png` - Eğitim grafikleri
- `confusion_matrix.png` - Karışıklık matrisi

## 🔧 Export

```python
from ultralytics_custom.model_builder import create_pothole_model

model = create_pothole_model()
model.export(format='onnx')  # ONNX export
model.export(format='torchscript')  # TorchScript export
```

## 📚 Referanslar

- [Dynamic Snake Convolution Paper](https://arxiv.org/abs/2307.08388)
- [SimAM Paper](https://proceedings.mlr.press/v139/yang21o.html)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

## 📝 Lisans

MIT License

## 🤝 Katkıda Bulunma

Pull request'ler kabul edilmektedir. Büyük değişiklikler için önce bir issue açın.

---

**Created with ❤️ for safer roads**

