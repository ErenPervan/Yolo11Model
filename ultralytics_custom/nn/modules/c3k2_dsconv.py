"""
C3k2 Block with DSConv and SimAM Integration
=============================================

YOLOv11'in C3k2 bloğunun DSConv ve SimAM entegreli versiyonu.
Bu blok, backbone ve neck'te kullanılarak çukur gibi düzensiz
şekillerin daha iyi algılanmasını sağlar.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .dsconv import DSConv, DySnakeConv
from .simam import SimAM
from .conv import ConvGELU, autopad


class Bottleneck_DSConv(nn.Module):
    """
    DSConv tabanlı Bottleneck bloğu
    
    Standart bottleneck'in DSConv versiyonu.
    Kıvrımlı yapıları daha iyi yakalar.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        shortcut (bool): Residual bağlantı kullanılsın mı
        g (int): Groups
        k (tuple): Kernel boyutları
        e (float): Expansion ratio
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[int, int] = (3, 3),
        e: float = 0.5
    ):
        super().__init__()
        
        c_ = int(c2 * e)  # Hidden channels
        
        # İlk konvolüsyon (1x1 veya 3x3)
        self.cv1 = ConvGELU(c1, c_, k[0], 1)
        
        # İkinci konvolüsyon DSConv ile
        self.cv2 = DySnakeConv(c_, c2, kernel_size=9)
        
        # SimAM attention
        self.attention = SimAM()
        
        # Shortcut bağlantı
        self.add = shortcut and c1 == c2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv1(x)
        out = self.cv2(out)
        out = self.attention(out)
        
        if self.add:
            out = out + x
        
        return out


class Bottleneck_SimAM(nn.Module):
    """
    SimAM tabanlı Bottleneck bloğu
    
    Standart bottleneck'e SimAM attention eklenmiş versiyonu.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        shortcut (bool): Residual bağlantı
        g (int): Groups
        k (tuple): Kernel boyutları
        e (float): Expansion ratio
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[int, int] = (3, 3),
        e: float = 0.5
    ):
        super().__init__()
        
        c_ = int(c2 * e)
        
        self.cv1 = ConvGELU(c1, c_, k[0], 1)
        self.cv2 = ConvGELU(c_, c2, k[1], 1, g=g)
        self.attention = SimAM()
        
        self.add = shortcut and c1 == c2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        out = self.attention(out)
        
        if self.add:
            out = out + x
        
        return out


class C3k2_DSConv(nn.Module):
    """
    C3k2 Block with DSConv Integration
    
    YOLOv11'in C3k2 bloğunun DSConv ve SimAM entegreli versiyonu.
    Split -> Bottleneck -> Concat yapısı korunmuş, DSConv eklenmiş.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        n (int): Bottleneck tekrar sayısı
        c3k (bool): C3k modunu kullan
        e (float): Expansion ratio
        g (int): Groups
        shortcut (bool): Residual bağlantı
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True
    ):
        super().__init__()
        
        self.c = int(c2 * e)  # Hidden channels
        
        # Giriş konvolüsyonları
        self.cv1 = ConvGELU(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvGELU((2 + n) * self.c, c2, 1, 1)
        
        # Bottleneck'ler (DSConv veya standart)
        if c3k:
            # DSConv tabanlı bottleneck'ler
            self.m = nn.ModuleList([
                Bottleneck_DSConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
                for _ in range(n)
            ])
        else:
            # SimAM tabanlı bottleneck'ler
            self.m = nn.ModuleList([
                Bottleneck_SimAM(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
                for _ in range(n)
            ])
        
        # Final attention
        self.attention = SimAM()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split
        y = list(self.cv1(x).chunk(2, dim=1))
        
        # Bottleneck'lerden geçir
        for m in self.m:
            y.append(m(y[-1]))
        
        # Concat ve çıkış
        out = self.cv2(torch.cat(y, dim=1))
        out = self.attention(out)
        
        return out


class C2f_DSConv(nn.Module):
    """
    C2f Block with DSConv
    
    YOLOv8/v11 C2f bloğunun DSConv versiyonu.
    Daha fazla gradient flow için tasarlanmış.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        n (int): Bottleneck sayısı
        shortcut (bool): Residual bağlantı
        g (int): Groups
        e (float): Expansion ratio
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5
    ):
        super().__init__()
        
        self.c = int(c2 * e)
        
        self.cv1 = ConvGELU(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvGELU((2 + n) * self.c, c2, 1, 1)
        
        self.m = nn.ModuleList([
            Bottleneck_DSConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class SPPF_SimAM(nn.Module):
    """
    Spatial Pyramid Pooling - Fast with SimAM
    
    YOLO'nun SPPF modülüne SimAM attention eklenmiş versiyonu.
    Multi-scale feature aggregation sağlar.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        k (int): Pooling kernel boyutu
    """
    
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        
        c_ = c1 // 2
        
        self.cv1 = ConvGELU(c1, c_, 1, 1)
        self.cv2 = ConvGELU(c_ * 4, c2, 1, 1)
        
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.attention = SimAM()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        
        out = self.cv2(torch.cat([x, y1, y2, y3], dim=1))
        out = self.attention(out)
        
        return out


class PSA_DSConv(nn.Module):
    """
    Position-Sensitive Attention with DSConv
    
    Pozisyon duyarlı attention mekanizması + DSConv.
    Segmentasyon için özellikle etkili.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        e (float): Expansion ratio
    """
    
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        
        c_ = int(c2 * e)
        
        # Feature processing
        self.cv1 = ConvGELU(c1, c_, 1, 1)
        self.cv2 = ConvGELU(c1, c_, 1, 1)
        
        # DSConv for spatial features
        self.dsconv = DySnakeConv(c_, c_, kernel_size=9)
        
        # Attention
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_, c_, 1),
            nn.GELU(),
            nn.Conv2d(c_, c_, 1),
            nn.Sigmoid()
        )
        
        # Output
        self.cv3 = ConvGELU(c_ * 2, c2, 1, 1)
        self.simam = SimAM()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.cv1(x)
        b = self.cv2(x)
        
        # DSConv branch
        a = self.dsconv(a)
        
        # Attention branch
        b = b * self.attn(b)
        
        # Concat ve output
        out = self.cv3(torch.cat([a, b], dim=1))
        out = self.simam(out)
        
        return out


class Detect_Head_Custom(nn.Module):
    """
    Custom Detection Head
    
    DSConv ve SimAM ile güçlendirilmiş detection head.
    Segmentasyon mask'ları için optimize edilmiş.
    
    Args:
        nc (int): Sınıf sayısı
        ch (tuple): Giriş kanalları
    """
    
    def __init__(self, nc: int = 80, ch: Tuple[int, ...] = ()):
        super().__init__()
        
        self.nc = nc
        self.nl = len(ch)  # Detection layer sayısı
        
        # Her scale için processing
        self.pre_process = nn.ModuleList([
            nn.Sequential(
                ConvGELU(c, c, 3, 1),
                SimAM()
            )
            for c in ch
        ])
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        return [pre(xi) for pre, xi in zip(self.pre_process, x)]

