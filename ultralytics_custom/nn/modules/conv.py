"""
Custom Convolution Modules with GELU Activation
================================================

GELU (Gaussian Error Linear Unit), varsayılan SiLU yerine kullanılarak
özellikle karmaşık desenlerde öğrenmenin daha stabil olması sağlanır.

GELU formülü:
    GELU(x) = x * Φ(x)
    
    Φ(x) = standart normal dağılımın CDF'i (Cumulative Distribution Function)
    
    Yaklaşık: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

SiLU vs GELU:
    - Her ikisi de smooth aktivasyon fonksiyonlarıdır
    - GELU, Transformer modellerinde yaygın olarak kullanılır (BERT, GPT)
    - GELU, SiLU'ya göre biraz daha smooth geçişler sağlar
    - Karmaşık pattern öğrenmede daha stabil gradyan akışı
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Union, Tuple


def autopad(k: Union[int, Tuple[int, int]], p: Optional[int] = None, d: int = 1):
    """
    Kernel boyutuna göre otomatik padding hesapla
    
    Args:
        k: Kernel size
        p: Padding (None ise otomatik hesapla)
        d: Dilation
    
    Returns:
        Padding değeri
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvGELU(nn.Module):
    """
    Standard Convolution with GELU activation
    
    Ultralytics Conv modülünün GELU aktivasyonlu versiyonu.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        k (int): Kernel boyutu (varsayılan 1)
        s (int): Stride (varsayılan 1)
        p (int, optional): Padding (None ise otomatik)
        g (int): Groups (varsayılan 1)
        d (int): Dilation (varsayılan 1)
        act (bool): Aktivasyon kullanılsın mı (varsayılan True)
    """
    
    default_act = nn.GELU()  # Varsayılan aktivasyon GELU
    
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        d: int = 1,
        act: Union[bool, nn.Module] = True
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            c1, c2, k, s, 
            padding=autopad(k, p, d), 
            groups=g, 
            dilation=d, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        
        # Aktivasyon seçimi
        if act is True:
            self.act = self.default_act
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Conv -> BN -> GELU"""
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Fused forward (Conv + Act, BN fused into Conv)"""
        return self.act(self.conv(x))


class DWConvGELU(nn.Module):
    """
    Depth-wise Convolution with GELU
    
    Her kanal için ayrı konvolüsyon uygular (groups=channels).
    Parametre sayısını azaltır.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        k (int): Kernel boyutu (varsayılan 1)
        s (int): Stride (varsayılan 1)
        d (int): Dilation (varsayılan 1)
        act (bool): Aktivasyon kullanılsın mı
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        d: int = 1,
        act: bool = True
    ):
        super().__init__()
        self.conv = ConvGELU(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvGELU_DSC(nn.Module):
    """
    Depthwise Separable Convolution with GELU
    
    Depthwise + Pointwise konvolüsyon kombinasyonu.
    Parametre verimliliği için kullanılır.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        k (int): Kernel boyutu
        s (int): Stride
    """
    
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        
        # Depthwise conv (her kanal için ayrı)
        self.dwconv = nn.Conv2d(
            c1, c1, k, s, 
            padding=autopad(k), 
            groups=c1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(c1)
        
        # Pointwise conv (1x1, kanal karıştırma)
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dwconv(x)))
        x = self.act(self.bn2(self.pwconv(x)))
        return x


class RepConvGELU(nn.Module):
    """
    Rep-style Convolution with GELU (Re-parameterization)
    
    Eğitim sırasında çoklu branch, inference sırasında tek branch.
    Daha güçlü öğrenme kapasitesi.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        k (int): Kernel boyutu
        s (int): Stride
    """
    
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        
        # Training branches
        self.conv1 = nn.Conv2d(c1, c2, k, s, padding=autopad(k), bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        
        self.conv2 = nn.Conv2d(c1, c2, 1, s, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        
        self.act = nn.GELU()
        
        # Identity branch (sadece c1 == c2 ve stride == 1 ise)
        self.identity = nn.BatchNorm2d(c1) if c1 == c2 and s == 1 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training forward (multi-branch)"""
        out = self.act(self.bn1(self.conv1(x)) + self.bn2(self.conv2(x)))
        
        if self.identity is not None:
            out = out + self.identity(x)
        
        return out


class GhostConvGELU(nn.Module):
    """
    Ghost Convolution with GELU
    
    Az sayıda filtre ile çok sayıda feature map üretir.
    Hesaplama verimliliği için kullanılır.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        k (int): Primary conv kernel boyutu
        s (int): Stride
        ratio (int): Ghost ratio
        dw_kernel (int): Depthwise kernel boyutu
    """
    
    def __init__(
        self, 
        c1: int, 
        c2: int, 
        k: int = 1, 
        s: int = 1, 
        ratio: int = 2,
        dw_kernel: int = 3
    ):
        super().__init__()
        
        c_ = c2 // ratio  # Primary feature sayısı
        
        # Primary convolution
        self.cv1 = ConvGELU(c1, c_, k, s)
        
        # Ghost (cheap) features
        self.cv2 = ConvGELU(c_, c_ * (ratio - 1), dw_kernel, 1, g=c_)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], dim=1)

