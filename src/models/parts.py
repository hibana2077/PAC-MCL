"""
Part extraction and pooling modules for PAC-MCL
Implements attention-guided part generation without requiring annotations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math


class SpatialAttention(nn.Module):
    """Spatial attention module for part discovery"""
    
    def __init__(self, in_channels: int, num_parts: int = 6):
        super().__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels
        
        # Attention layers
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv2 = nn.Conv2d(in_channels // 4, num_parts, 1)
        
        # Layer normalization
        self.norm = nn.GroupNorm(1, in_channels // 4)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate attention maps for parts
        
        Args:
            x: Feature tensor [B, C, H, W]
            
        Returns:
            Attention maps [B, num_parts, H, W]
        """
        # Generate attention maps
        att = F.relu(self.norm(self.conv1(x)))
        att = self.conv2(att)
        
        # Apply spatial softmax
        B, P, H, W = att.shape
        att = att.view(B, P, -1)
        att = F.softmax(att, dim=-1)
        att = att.view(B, P, H, W)
        
        return att


class PartExtractor(nn.Module):
    """Extract parts from feature maps using attention"""
    
    def __init__(self, 
                 in_channels: int,
                 num_parts: int = 6,
                 part_dim: int = 64,
                 use_attention: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_parts = num_parts
        self.part_dim = part_dim
        self.use_attention = use_attention
        
        # Channel adaptation layer (will be created dynamically if needed)
        self.channel_adapter = None
        
        # Spatial attention for part discovery
        if use_attention:
            self.attention = SpatialAttention(in_channels, num_parts)
        
        # Dimension reduction
        self.dim_reduction = nn.Sequential(
            nn.Conv2d(in_channels, part_dim, 1, bias=False),
            nn.BatchNorm2d(part_dim),
            nn.ReLU(inplace=True)
        )
        
        # Alternative: learnable part prototypes
        if not use_attention:
            self.part_prototypes = nn.Parameter(torch.randn(num_parts, in_channels))
            nn.init.kaiming_normal_(self.part_prototypes)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract parts from feature maps
        
        Args:
            x: Feature tensor [B, C, H, W] for CNN or [B, seq_len, embed_dim] for ViT
            
        Returns:
            Dict containing:
                - 'part_features': List of part feature tensors
                - 'attention_maps': Attention maps [B, P, H, W] (if using attention)
                - 'spatial_features': Reduced spatial features [B, part_dim, H, W]
        """
        # Handle both CNN (4D) and ViT (3D) inputs
        if x.dim() == 3:
            # ViT input: [B, seq_len, embed_dim]
            B, seq_len, embed_dim = x.shape
            # Convert to spatial format for compatibility
            # Assume square spatial arrangement (patch grid)
            H = W = int(seq_len ** 0.5)
            if H * W != seq_len:
                # Handle non-square arrangements or add CLS token
                # For ViT with CLS token, remove it
                if seq_len == H * W + 1:
                    x = x[:, 1:, :]  # Remove CLS token
                    seq_len -= 1
                    H = W = int(seq_len ** 0.5)
                else:
                    # Fallback: pad to nearest square
                    H = W = int(seq_len ** 0.5) + 1
                    pad_len = H * W - seq_len
                    if pad_len > 0:
                        padding = torch.zeros(B, pad_len, embed_dim, device=x.device, dtype=x.dtype)
                        x = torch.cat([x, padding], dim=1)
            
            # Reshape to spatial format: [B, embed_dim, H, W]
            x = x.transpose(1, 2).reshape(B, embed_dim, H, W)
            C = embed_dim
        elif x.dim() == 4:
            # CNN input: [B, C, H, W]
            B, C, H, W = x.shape
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D tensor with shape {x.shape}")
        
        # Handle channel mismatch with dynamic adapter
        if C != self.in_channels:
            if self.channel_adapter is None:
                # Create channel adapter on first use
                self.channel_adapter = nn.Conv2d(C, self.in_channels, 1, bias=False).to(x.device)
                nn.init.kaiming_normal_(self.channel_adapter.weight)
            x = self.channel_adapter(x)
            C = self.in_channels
        
        # Reduce dimensions first
        spatial_features = self.dim_reduction(x)  # [B, part_dim, H, W]
        
        if self.use_attention:
            # Generate attention maps
            attention_maps = self.attention(x)  # [B, num_parts, H, W]
            
            # Extract part features using attention
            part_features = []
            for p in range(self.num_parts):
                att_map = attention_maps[:, p:p+1, :, :]  # [B, 1, H, W]
                
                # Weighted pooling
                weighted_features = spatial_features * att_map  # [B, part_dim, H, W]
                
                # Global average pooling with attention weights
                att_weights = att_map.view(B, 1, -1)  # [B, 1, H*W]
                spatial_flat = weighted_features.view(B, self.part_dim, -1)  # [B, part_dim, H*W]
                
                # Weighted average
                part_feat = torch.sum(spatial_flat * att_weights, dim=2) / (torch.sum(att_weights, dim=2) + 1e-8)
                part_features.append(part_feat)  # [B, part_dim]
            
            return {
                'part_features': part_features,
                'attention_maps': attention_maps,
                'spatial_features': spatial_features
            }
        
        else:
            # Use learnable prototypes with similarity matching
            spatial_flat = spatial_features.view(B, self.part_dim, -1)  # [B, part_dim, H*W]
            
            part_features = []
            similarity_maps = []
            
            for p in range(self.num_parts):
                prototype = self.part_prototypes[p]  # [in_channels]
                
                # Compute similarity with original features
                x_flat = x.view(B, C, -1)  # [B, C, H*W]
                similarity = torch.matmul(prototype.unsqueeze(0), x_flat)  # [B, 1, H*W]
                similarity = F.softmax(similarity, dim=-1)
                similarity_maps.append(similarity.view(B, 1, H, W))
                
                # Extract part features
                part_feat = torch.sum(spatial_flat * similarity, dim=2)  # [B, part_dim]
                part_features.append(part_feat)
            
            attention_maps = torch.cat(similarity_maps, dim=1)  # [B, num_parts, H, W]
            
            return {
                'part_features': part_features,
                'attention_maps': attention_maps,
                'spatial_features': spatial_features
            }


class PartPooling(nn.Module):
    """Advanced part pooling with multiple strategies"""
    
    def __init__(self, 
                 part_dim: int = 64,
                 pooling_method: str = 'attention',
                 temperature: float = 1.0):
        super().__init__()
        
        self.part_dim = part_dim
        self.pooling_method = pooling_method
        self.temperature = temperature
        
        if pooling_method == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(part_dim, part_dim // 4),
                nn.ReLU(),
                nn.Linear(part_dim // 4, 1)
            )
    
    def pool_part_tokens(self, 
                        tokens: torch.Tensor, 
                        attention_map: torch.Tensor) -> torch.Tensor:
        """
        Pool spatial tokens to form part representation
        
        Args:
            tokens: Spatial feature tokens [B, H*W, part_dim]
            attention_map: Attention weights [B, H*W]
            
        Returns:
            Part representation [B, part_dim]
        """
        if self.pooling_method == 'weighted_avg':
            # Weighted average pooling
            weights = F.softmax(attention_map / self.temperature, dim=1)  # [B, H*W]
            part_repr = torch.sum(tokens * weights.unsqueeze(-1), dim=1)  # [B, part_dim]
            
        elif self.pooling_method == 'attention':
            # Learnable attention pooling
            att_scores = self.attention_pooling(tokens).squeeze(-1)  # [B, H*W]
            att_scores = att_scores + attention_map  # Add spatial attention
            weights = F.softmax(att_scores / self.temperature, dim=1)
            part_repr = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
            
        elif self.pooling_method == 'max':
            # Max pooling with attention masking
            masked_tokens = tokens * attention_map.unsqueeze(-1)
            part_repr, _ = torch.max(masked_tokens, dim=1)
            
        else:  # 'avg'
            # Simple average pooling
            part_repr = torch.mean(tokens, dim=1)
        
        return part_repr
    
    def forward(self, 
                spatial_features: torch.Tensor,
                attention_maps: torch.Tensor) -> List[torch.Tensor]:
        """
        Pool parts from spatial features using attention maps
        
        Args:
            spatial_features: Spatial features [B, part_dim, H, W]
            attention_maps: Attention maps [B, num_parts, H, W]
            
        Returns:
            List of part features [B, part_dim] for each part
        """
        B, part_dim, H, W = spatial_features.shape
        _, num_parts, _, _ = attention_maps.shape
        
        # Reshape to tokens
        tokens = spatial_features.view(B, part_dim, -1).permute(0, 2, 1)  # [B, H*W, part_dim]
        
        part_features = []
        for p in range(num_parts):
            att_map = attention_maps[:, p, :, :].view(B, -1)  # [B, H*W]
            part_feat = self.pool_part_tokens(tokens, att_map)
            part_features.append(part_feat)
        
        return part_features


class AdaptivePartExtractor(nn.Module):
    """
    Adaptive part extractor that can handle different input resolutions
    and dynamically adjust the number of parts
    """
    
    def __init__(self,
                 in_channels: int,
                 base_num_parts: int = 6,
                 part_dim: int = 64,
                 adaptive_parts: bool = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_num_parts = base_num_parts
        self.part_dim = part_dim
        self.adaptive_parts = adaptive_parts
        
        # Base part extractor
        self.part_extractor = PartExtractor(
            in_channels=in_channels,
            num_parts=base_num_parts,
            part_dim=part_dim,
            use_attention=True
        )
        
        # Adaptive components
        if adaptive_parts:
            self.part_selector = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, base_num_parts),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract parts with optional adaptive selection
        
        Args:
            x: Feature tensor [B, C, H, W]
            
        Returns:
            Dict containing extracted parts and attention maps
        """
        # Extract base parts
        part_output = self.part_extractor(x)
        
        if self.adaptive_parts:
            # Compute part importance scores
            part_scores = self.part_selector(x)  # [B, base_num_parts]
            
            # Filter parts based on importance
            part_features = part_output['part_features']
            attention_maps = part_output['attention_maps']
            
            # Apply importance weighting
            weighted_parts = []
            for i, part_feat in enumerate(part_features):
                weight = part_scores[:, i:i+1]  # [B, 1]
                weighted_part = part_feat * weight
                weighted_parts.append(weighted_part)
            
            part_output['part_features'] = weighted_parts
            part_output['part_importance'] = part_scores
        
        return part_output


if __name__ == "__main__":
    # Test part extraction
    print("Testing part extraction modules...")
    
    # Test parameters
    batch_size = 2
    in_channels = 512
    height, width = 14, 14
    num_parts = 6
    part_dim = 64
    
    # Create dummy input
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"Input shape: {x.shape}")
    
    # Test PartExtractor
    print("\n=== Testing PartExtractor ===")
    part_extractor = PartExtractor(
        in_channels=in_channels,
        num_parts=num_parts,
        part_dim=part_dim,
        use_attention=True
    )
    
    output = part_extractor(x)
    print(f"Number of parts: {len(output['part_features'])}")
    print(f"Part feature shape: {output['part_features'][0].shape}")
    print(f"Attention maps shape: {output['attention_maps'].shape}")
    print(f"Spatial features shape: {output['spatial_features'].shape}")
    
    # Test PartPooling
    print("\n=== Testing PartPooling ===")
    part_pooling = PartPooling(part_dim=part_dim, pooling_method='attention')
    
    pooled_parts = part_pooling(
        output['spatial_features'],
        output['attention_maps']
    )
    print(f"Number of pooled parts: {len(pooled_parts)}")
    print(f"Pooled part shape: {pooled_parts[0].shape}")
    
    # Test AdaptivePartExtractor
    print("\n=== Testing AdaptivePartExtractor ===")
    adaptive_extractor = AdaptivePartExtractor(
        in_channels=in_channels,
        base_num_parts=num_parts,
        part_dim=part_dim,
        adaptive_parts=True
    )
    
    adaptive_output = adaptive_extractor(x)
    print(f"Adaptive parts: {len(adaptive_output['part_features'])}")
    if 'part_importance' in adaptive_output:
        print(f"Part importance shape: {adaptive_output['part_importance'].shape}")
    
    print("\nAll tests passed!")
