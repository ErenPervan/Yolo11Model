"""
Pothole Segmentation Model - Training Script
=============================================

Bu script, DSConv + SimAM + GELU entegreli çukur segmentasyon
modelini eğitmek için kullanılır.

Yerel veya Colab'da çalıştırılabilir.

Kullanım:
    python train.py --data pothole.yaml --epochs 100 --batch 16
"""

import argparse
import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO

# Custom modülleri import et
from ultralytics_custom.nn.modules import DSConv, DySnakeConv, SimAM, ConvGELU, C3k2_DSConv
from ultralytics_custom.model_builder import (
    register_custom_modules,
    PotholeSegmentationModel,
    create_pothole_model
)


def parse_args():
    """Komut satırı argümanlarını parse et"""
    parser = argparse.ArgumentParser(
        description='Pothole Segmentation Model Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Temel parametreler
    parser.add_argument('--data', type=str, required=True,
                        help='Veri seti YAML dosyası yolu')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Eğitim epoch sayısı')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Görüntü boyutu')
    
    # Model parametreleri
    parser.add_argument('--model', type=str, default='yolo11s-seg.pt',
                        help='Başlangıç modeli (pretrained)')
    parser.add_argument('--custom-cfg', type=str, default=None,
                        help='Custom model YAML konfigürasyonu')
    
    # Eğitim parametreleri
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Başlangıç learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='Final learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=float, default=3.0,
                        help='Warmup epoch sayısı')
    
    # Augmentation parametreleri
    parser.add_argument('--hsv-h', type=float, default=0.015,
                        help='HSV-Hue augmentation')
    parser.add_argument('--hsv-s', type=float, default=0.7,
                        help='HSV-Saturation augmentation')
    parser.add_argument('--hsv-v', type=float, default=0.4,
                        help='HSV-Value augmentation')
    parser.add_argument('--degrees', type=float, default=0.0,
                        help='Rotasyon derecesi')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='Translation augmentation')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Scale augmentation')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='Horizontal flip olasılığı')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic augmentation')
    
    # Diğer parametreler
    parser.add_argument('--device', type=str, default='',
                        help='Cihaz (cuda:0, cpu, vs.)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Dataloader worker sayısı')
    parser.add_argument('--project', type=str, default='runs/pothole_seg',
                        help='Proje dizini')
    parser.add_argument('--name', type=str, default='train',
                        help='Deney adı')
    parser.add_argument('--exist-ok', action='store_true',
                        help='Mevcut proje dizinini üzerine yaz')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Pretrained ağırlık kullan')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'Adam', 'AdamW', 'RMSProp'],
                        help='Optimizer')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume training from checkpoint')
    parser.add_argument('--modify-act', action='store_true', default=True,
                        help='Aktivasyonları GELU\'ya çevir')
    
    return parser.parse_args()


def print_banner():
    """Başlık banner'ı yazdır"""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║      🕳️  POTHOLE SEGMENTATION MODEL TRAINING  🕳️               ║
║                                                                ║
║      DSConv + SimAM + GELU Enhanced YOLOv11                   ║
╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config(args):
    """Konfigürasyonu yazdır"""
    print("\n📋 Eğitim Konfigürasyonu:")
    print("-" * 40)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 40)


def check_environment():
    """Ortamı kontrol et"""
    print("\n🔍 Ortam Kontrolü:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Python: {sys.version.split()[0]}")


def main():
    """Ana eğitim fonksiyonu"""
    print_banner()
    
    # Argümanları parse et
    args = parse_args()
    print_config(args)
    
    # Ortam kontrolü
    check_environment()
    
    # Custom modülleri kaydet
    print("\n📦 Custom modüller kaydediliyor...")
    register_custom_modules()
    
    # Model oluştur
    print("\n🏗️ Model oluşturuluyor...")
    
    if args.custom_cfg:
        # Custom config kullan
        model = PotholeSegmentationModel(
            model_cfg=args.custom_cfg,
            task='segment'
        )
    else:
        # Pretrained model kullan
        model = PotholeSegmentationModel(
            pretrained=args.model,
            task='segment'
        )
    
    # Aktivasyonları modifiye et
    if args.modify_act:
        model.modify_with_custom_modules()
    
    # Model bilgisi
    print(f"\n📊 Model Bilgisi:")
    print(f"  Task: {model.task}")
    print(f"  Config: {args.custom_cfg or args.model}")
    
    # Eğitimi başlat
    print("\n🚀 Eğitim başlatılıyor...")
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        device=args.device if args.device else None,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        verbose=args.verbose,
        seed=args.seed,
        resume=args.resume if args.resume else False,
    )
    
    print("\n✅ Eğitim tamamlandı!")
    print(f"📁 Sonuçlar: {args.project}/{args.name}")
    
    return results


if __name__ == '__main__':
    main()

