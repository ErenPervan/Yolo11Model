"""
Custom Model Builder
====================

Bu dosya, custom modülleri Ultralytics YOLO sistemine entegre eder.
Modeli oluşturmak ve eğitmek için bu builder'ı kullanın.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union
import yaml

# Ultralytics imports
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.modules import (
    Conv, Concat, Segment, SPPF, C2PSA, C3k2
)

# Custom modüller
from ultralytics_custom.nn.modules import (
    DSConv, DySnakeConv, SimAM, ConvGELU, C3k2_DSConv
)
from ultralytics_custom.nn.modules.c3k2_dsconv import (
    SPPF_SimAM, PSA_DSConv, Bottleneck_DSConv, Bottleneck_SimAM,
    C2f_DSConv
)


# Custom modülleri global namespace'e ekle
CUSTOM_MODULES = {
    'DSConv': DSConv,
    'DySnakeConv': DySnakeConv,
    'SimAM': SimAM,
    'ConvGELU': ConvGELU,
    'C3k2_DSConv': C3k2_DSConv,
    'SPPF_SimAM': SPPF_SimAM,
    'PSA_DSConv': PSA_DSConv,
    'Bottleneck_DSConv': Bottleneck_DSConv,
    'Bottleneck_SimAM': Bottleneck_SimAM,
    'C2f_DSConv': C2f_DSConv,
}


def register_custom_modules():
    """
    Custom modülleri Ultralytics sistemine kaydet.
    Bu fonksiyonu model oluşturmadan önce çağırın.
    """
    import ultralytics.nn.modules as modules
    
    for name, module in CUSTOM_MODULES.items():
        setattr(modules, name, module)
        print(f"✓ {name} modülü kaydedildi")
    
    # tasks.py'deki modül listesine de ekle
    try:
        from ultralytics.nn import tasks
        if hasattr(tasks, 'CUSTOM_MODULES'):
            tasks.CUSTOM_MODULES.update(CUSTOM_MODULES)
        print("✓ Tüm custom modüller başarıyla kaydedildi!")
    except Exception as e:
        print(f"⚠ tasks modülüne ekleme yapılamadı: {e}")


class PotholeSegmentationModel:
    """
    Çukur Segmentasyon Modeli
    
    DSConv, SimAM ve GELU ile güçlendirilmiş YOLOv11 tabanlı
    segmentasyon modeli.
    
    Kullanım:
        ```python
        model = PotholeSegmentationModel()
        model.train(data='pothole.yaml', epochs=100)
        ```
    """
    
    def __init__(
        self,
        model_cfg: Optional[str] = None,
        pretrained: Optional[str] = None,
        task: str = 'segment'
    ):
        """
        Args:
            model_cfg: Model YAML konfigürasyon dosyası yolu
            pretrained: Pretrained ağırlık dosyası (yolo11s-seg.pt gibi)
            task: Görev tipi ('segment' veya 'detect')
        """
        # Custom modülleri kaydet
        register_custom_modules()
        
        self.task = task
        self.model_cfg = model_cfg
        
        # Model oluştur
        if pretrained:
            # Pretrained modelden başla
            self.model = YOLO(pretrained)
            print(f"✓ Pretrained model yüklendi: {pretrained}")
        elif model_cfg:
            # Custom config'den oluştur
            self.model = YOLO(model_cfg, task=task)
            print(f"✓ Custom model oluşturuldu: {model_cfg}")
        else:
            # Varsayılan YOLOv11s-seg
            self.model = YOLO('yolo11s-seg.pt')
            print("✓ Varsayılan yolo11s-seg modeli yüklendi")
    
    def modify_with_custom_modules(self):
        """
        Mevcut modeli custom modüllerle modifiye et.
        Aktivasyon fonksiyonlarını GELU'ya çevir ve SimAM ekle.
        """
        print("\n🔧 Model modifikasyonu başlatılıyor...")
        
        model = self.model.model
        
        modifications = 0
        
        # Tüm modülleri dolaş
        for name, module in model.named_modules():
            # SiLU -> GELU değişimi
            if isinstance(module, nn.SiLU):
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                    setattr(parent, attr_name, nn.GELU())
                    modifications += 1
        
        print(f"✓ {modifications} aktivasyon fonksiyonu GELU'ya çevrildi")
        return self
    
    def train(
        self,
        data: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        **kwargs
    ):
        """
        Modeli eğit.
        
        Args:
            data: Veri seti YAML dosyası
            epochs: Epoch sayısı
            imgsz: Görüntü boyutu
            batch: Batch size
            **kwargs: Diğer eğitim parametreleri
        """
        return self.model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            **kwargs
        )
    
    def val(self, **kwargs):
        """Validasyon çalıştır"""
        return self.model.val(**kwargs)
    
    def predict(self, source, **kwargs):
        """Tahmin yap"""
        return self.model.predict(source, **kwargs)
    
    def export(self, format: str = 'onnx', **kwargs):
        """Modeli dışa aktar"""
        return self.model.export(format=format, **kwargs)
    
    def save(self, path: str):
        """Modeli kaydet"""
        torch.save(self.model.model.state_dict(), path)
        print(f"✓ Model kaydedildi: {path}")
    
    def load(self, path: str):
        """Model ağırlıklarını yükle"""
        self.model.model.load_state_dict(torch.load(path))
        print(f"✓ Ağırlıklar yüklendi: {path}")


def create_pothole_model(
    use_pretrained: bool = True,
    modify_activations: bool = True
) -> PotholeSegmentationModel:
    """
    Çukur tespiti için hazır model oluştur.
    
    Args:
        use_pretrained: Pretrained ağırlık kullanılsın mı
        modify_activations: Aktivasyonlar GELU'ya çevrilsin mi
    
    Returns:
        PotholeSegmentationModel instance
    """
    if use_pretrained:
        model = PotholeSegmentationModel(pretrained='yolo11s-seg.pt')
    else:
        cfg_path = Path(__file__).parent / 'cfg' / 'models' / 'pothole_seg.yaml'
        model = PotholeSegmentationModel(model_cfg=str(cfg_path))
    
    if modify_activations:
        model.modify_with_custom_modules()
    
    return model


if __name__ == '__main__':
    # Test
    print("=" * 50)
    print("Custom Pothole Segmentation Model Builder")
    print("=" * 50)
    
    # Modülleri kaydet
    register_custom_modules()
    
    # Model bilgisi
    print("\n📦 Mevcut custom modüller:")
    for name in CUSTOM_MODULES:
        print(f"  - {name}")

