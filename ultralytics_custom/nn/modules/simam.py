"""
Simple Attention Module (SimAM)
================================

SimAM, parametre eklemeden çalışan bir dikkat mekanizmasıdır.
Her nöronun önemini, o nöronun diğer nöronlardan ne kadar farklı olduğuna 
göre hesaplar. Bu sayede model, önemli piksellere (örn. çukur içi) 
odaklanıp arka planı (sağlam yol) daha iyi görmezden gelir.

Paper: "SimAM: A Simple, Parameter-Free Attention Module for 
        Convolutional Neural Networks"
        ICML 2021
        https://proceedings.mlr.press/v139/yang21o.html
"""

import torch
import torch.nn as nn
from typing import Optional


class SimAM(nn.Module):
    """
    Simple Attention Module (SimAM)
    
    Parametre-free dikkat mekanizması. Her nöronun önemini, o nöronun
    spatial boyuttaki diğer nöronlardan ne kadar ayrıştığına göre hesaplar.
    
    Energy function:
        e_t = (x_t - μ)² / (4 * (σ² + λ))
    
    Attention weight:
        y = x * sigmoid(e)
    
    Args:
        e_lambda (float): Numerical stability için epsilon değeri (varsayılan 1e-4)
    
    Note:
        Bu modül hiç eğitilebilir parametre içermez!
        Bu sayede overfitting riski azalır ve hesaplama maliyeti düşer.
    """
    
    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Attention-weighted tensor [B, C, H, W]
        """
        # Batch size, Channel, Height, Width
        B, C, H, W = x.shape
        
        # Spatial boyut (n = H * W - 1)
        n = H * W - 1
        
        # Her kanal için mean hesapla
        # x_minus_mu: (x - μ)
        x_minus_mu = x - x.mean(dim=[2, 3], keepdim=True)
        
        # Variance hesapla: σ² = Σ(x - μ)² / n
        # x_minus_mu_sq: (x - μ)²
        x_minus_mu_sq = x_minus_mu.pow(2)
        
        # Variance (spatial boyutta)
        variance = x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n
        
        # Energy hesapla: e = (x - μ)² / (4 * (σ² + λ))
        # Düşük energy = daha önemli nöron (diğerlerinden farklı)
        energy = x_minus_mu_sq / (4 * (variance + self.e_lambda) + self.e_lambda)
        
        # Attention weight: sigmoid(energy)
        # Yüksek energy -> yüksek attention (farklı nöronlara daha fazla dikkat)
        attention = torch.sigmoid(energy)
        
        # Attention uygula
        out = x * attention
        
        return out
    
    def extra_repr(self) -> str:
        return f'e_lambda={self.e_lambda}'


class SimAM_Enhanced(nn.Module):
    """
    Enhanced SimAM with channel attention
    
    Standart SimAM'a ek olarak kanal bazlı dikkat de ekler.
    Bu versiyon biraz daha ağır ama daha güçlü sonuçlar verebilir.
    
    Args:
        channels (int): Kanal sayısı (opsiyonel, channel attention için)
        e_lambda (float): Numerical stability için epsilon
        use_channel_att (bool): Kanal attention kullanılsın mı
    """
    
    def __init__(
        self,
        channels: Optional[int] = None,
        e_lambda: float = 1e-4,
        use_channel_att: bool = False
    ):
        super().__init__()
        self.e_lambda = e_lambda
        self.use_channel_att = use_channel_att
        
        if use_channel_att and channels is not None:
            self.channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels // 4),
                nn.GELU(),
                nn.Linear(channels // 4, channels),
                nn.Sigmoid()
            )
        else:
            self.channel_att = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        B, C, H, W = x.shape
        n = H * W - 1
        
        # Spatial attention (SimAM)
        x_minus_mu = x - x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_sq = x_minus_mu.pow(2)
        variance = x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n
        energy = x_minus_mu_sq / (4 * (variance + self.e_lambda) + self.e_lambda)
        spatial_att = torch.sigmoid(energy)
        
        out = x * spatial_att
        
        # Channel attention (opsiyonel)
        if self.channel_att is not None:
            channel_weights = self.channel_att(x)  # [B, C]
            channel_weights = channel_weights.view(B, C, 1, 1)  # [B, C, 1, 1]
            out = out * channel_weights
        
        return out


class SimAM_Block(nn.Module):
    """
    SimAM Block - Konvolüsyon + SimAM kombinasyonu
    
    Ultralytics modeline kolay entegrasyon için tasarlanmış blok.
    
    Args:
        c1 (int): Giriş kanal sayısı
        c2 (int): Çıkış kanal sayısı
        e_lambda (float): SimAM epsilon değeri
    """
    
    def __init__(self, c1: int, c2: int, e_lambda: float = 1e-4):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )
        self.simam = SimAM(e_lambda)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.simam(x)
        return x

