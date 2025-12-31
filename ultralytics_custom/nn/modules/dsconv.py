"""
Dynamic Snake Convolution (DSConv) Module
==========================================

DSConv, standart konvolüsyonların aksine kernel yapısını hedef nesnenin şekline
dinamik olarak hizalar. Bu sayede kıvrımlı ve düzensiz kenarlı nesnelerin 
(örneğin çukurlar) segmentasyonunda daha hassas sonuçlar elde edilir.

Paper: "Dynamic Snake Convolution based on Topological Geometric Constraints 
        for Tubular Structure Segmentation"
        https://arxiv.org/abs/2307.08388
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DSConv(nn.Module):
    """
    Dynamic Snake Convolution (DSConv)
    
    Standart konvolüsyonun kernel pozisyonlarını dinamik olarak ayarlayarak
    kıvrımlı/yılan şeklindeki yapıları daha iyi yakalar.
    
    Args:
        in_channels (int): Giriş kanal sayısı
        out_channels (int): Çıkış kanal sayısı
        kernel_size (int): Kernel boyutu (varsayılan 9 - yılan uzunluğu)
        extend_scope (float): Offset genişleme faktörü (varsayılan 1.0)
        morph (int): Morfoji yönü - 0: x ekseni, 1: y ekseni
        if_offset (bool): Offset öğrenip öğrenmeyeceği
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        
        # Offset öğrenmek için konvolüsyon katmanı
        # Her pozisyon için bir offset değeri (kernel_size kadar)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size,  # x ve y offset'leri
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        # BatchNorm offset'ler için
        self.bn_offset = nn.BatchNorm2d(2 * kernel_size)
        
        # Ana konvolüsyon katmanı
        # Yılan şeklinde örneklenen pikseller için 1D konvolüsyon mantığı
        if morph == 0:  # x ekseni boyunca
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
                bias=False
            )
        else:  # y ekseni boyunca
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=1,
                padding=(kernel_size // 2, 0),
                bias=False
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()  # GELU aktivasyon
        
        self._init_weights()
    
    def _init_weights(self):
        """Ağırlıkları başlat"""
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
    
    def _get_offset(self, x: torch.Tensor) -> torch.Tensor:
        """Offset değerlerini hesapla"""
        offset = self.offset_conv(x)
        offset = self.bn_offset(offset)
        offset = torch.tanh(offset)  # [-1, 1] aralığına normalize et
        offset = offset * self.extend_scope
        return offset
    
    def _deform_sample(
        self, 
        x: torch.Tensor, 
        offset: torch.Tensor
    ) -> torch.Tensor:
        """
        Dinamik yılan şeklinde örnekleme yap
        
        Args:
            x: Giriş feature map [B, C, H, W]
            offset: Offset değerleri [B, 2*kernel_size, H, W]
        
        Returns:
            Deformed feature map
        """
        B, C, H, W = x.shape
        
        # Grid oluştur
        device = x.device
        dtype = x.dtype
        
        # Normalize grid koordinatları [-1, 1]
        y_range = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x_range = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
        
        # Grid'i batch boyutuna genişlet
        base_grid = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        
        if self.if_offset:
            # Offset'leri uygula
            # offset: [B, 2*K, H, W] -> sadece merkez offset kullan
            offset_x = offset[:, :self.kernel_size, :, :].mean(dim=1, keepdim=True)  # [B, 1, H, W]
            offset_y = offset[:, self.kernel_size:, :, :].mean(dim=1, keepdim=True)  # [B, 1, H, W]
            
            # Offset'leri normalize et
            offset_x = offset_x.permute(0, 2, 3, 1) * 2.0 / W  # [B, H, W, 1]
            offset_y = offset_y.permute(0, 2, 3, 1) * 2.0 / H  # [B, H, W, 1]
            
            offset_combined = torch.cat([offset_x, offset_y], dim=-1)  # [B, H, W, 2]
            
            # Deformed grid
            deformed_grid = base_grid + offset_combined
            
            # Grid sample ile örnekle
            x_deformed = F.grid_sample(
                x, 
                deformed_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
        else:
            x_deformed = x
        
        return x_deformed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Offset hesapla
        if self.if_offset:
            offset = self._get_offset(x)
            x = self._deform_sample(x, offset)
        
        # Yılan konvolüsyonu uygula
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        
        return x


class DySnakeConv(nn.Module):
    """
    Dynamic Snake Convolution Block
    
    İki farklı yönde (x ve y ekseni) DSConv uygular ve birleştirir.
    Bu sayede her iki yöndeki kıvrımlı yapılar da yakalanır.
    
    Args:
        in_channels (int): Giriş kanal sayısı
        out_channels (int): Çıkış kanal sayısı
        kernel_size (int): Kernel boyutu
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9
    ):
        super().__init__()
        
        # Standart 3x3 konvolüsyon
        self.conv_0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # X ekseni boyunca yılan konvolüsyonu
        self.conv_x = DSConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            morph=0,
            if_offset=True
        )
        
        # Y ekseni boyunca yılan konvolüsyonu
        self.conv_y = DSConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            morph=1,
            if_offset=True
        )
        
        # Birleştirme için 1x1 konvolüsyon
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Üç farklı konvolüsyon uygula
        out_0 = self.conv_0(x)
        out_x = self.conv_x(x)
        out_y = self.conv_y(x)
        
        # Birleştir
        out = torch.cat([out_0, out_x, out_y], dim=1)
        out = self.conv_out(out)
        
        return out


class DSConv_Simple(nn.Module):
    """
    Basitleştirilmiş Dynamic Snake Convolution
    
    Daha hafif bir versiyon - sadece tek yönde çalışır ve daha az parametre.
    Ultralytics modeline entegrasyon için uygun.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        k (int): Kernel boyutu (varsayılan 3)
        s (int): Stride (varsayılan 1)
    """
    
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        
        self.c1 = c1
        self.c2 = c2
        
        # Offset öğrenme katmanı
        self.offset_conv = nn.Conv2d(c1, 2 * k * k, kernel_size=3, padding=1, bias=True)
        
        # Ana konvolüsyon
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU()
        
        self._init_offset()
    
    def _init_offset(self):
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Basit forward - offset sadece attention benzeri çalışır
        offset = torch.tanh(self.offset_conv(x))
        
        # Offset'i attention weight olarak kullan
        offset_weight = offset.mean(dim=1, keepdim=True)
        offset_weight = torch.sigmoid(offset_weight)
        
        # Ana konvolüsyon
        out = self.conv(x)
        out = self.bn(out)
        
        # Offset attention uygula
        out = out * offset_weight
        out = self.act(out)
        
        return out

