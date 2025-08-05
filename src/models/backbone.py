"""
Timm backbone wrapper for PAC-MCL
Provides a unified interface to various CNN architectures from timm
"""

import torch
import torch.nn as nn
import timm
import timm.data
from typing import Dict, Any, List, Optional


class TimmBackbone(nn.Module):
    """
    Wrapper around timm models to extract features at multiple levels
    """
    
    def __init__(self, 
                 model_name: str = 'resnet50',
                 pretrained: bool = True,
                 num_classes: int = 1000,
                 feature_levels: List[str] = None,
                 freeze_backbone: bool = False):
        """
        Args:
            model_name: Name of the timm model
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes
            feature_levels: List of feature levels to extract
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Create timm model
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get model info
        self.model_info = timm.models.get_pretrained_cfg(model_name)
        
        # Set feature levels
        if feature_levels is None:
            # Default feature extraction points
            feature_levels = self._get_default_feature_levels()
        self.feature_levels = feature_levels
        
        # Create feature hooks if needed
        self.feature_hooks = {}
        self.feature_outputs = {}
        self._register_hooks()
        
        # Classification head
        feature_dim = self._get_feature_dim()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
    
    def _get_device(self) -> torch.device:
        """Get the device of the model"""
        return next(self.model.parameters()).device
    
    def _get_input_size(self) -> tuple:
        """Get the proper input size for this model"""
        try:
            # Get data config from pretrained configuration
            data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
            input_size = data_cfg['input_size']
            return input_size
        except (AttributeError, KeyError):
            # Fallback to default size if config is not available
            return (3, 224, 224)
    
    def _get_default_feature_levels(self) -> List[str]:
        """Get default feature extraction levels based on model type"""
        if 'resnet' in self.model_name.lower():
            return ['layer3', 'layer4']
        elif 'efficientnet' in self.model_name.lower():
            return ['blocks.5', 'blocks.7']
        elif 'convnext' in self.model_name.lower():
            return ['stages.2', 'stages.3']
        elif 'regnet' in self.model_name.lower():
            return ['s3', 's4']
        else:
            # Generic approach - try to find last few layers
            return ['features']
    
    def _get_feature_dim(self) -> int:
        """Get the feature dimension of the backbone"""
        # Get proper input size for this model
        input_size = self._get_input_size()
        # Create dummy input to infer dimensions
        dummy_input = torch.randn(1, *input_size, device=self._get_device())
        with torch.no_grad():
            features = self.model(dummy_input)
            if isinstance(features, dict):
                # Multi-level features
                last_features = list(features.values())[-1]
            else:
                last_features = features
            
            # Handle different tensor shapes
            if 'vit' in self.model_name.lower() or 'deit' in self.model_name.lower():
                # Vision Transformer - features are [B, seq_len, embed_dim]
                if last_features.dim() == 3:
                    return last_features.shape[-1]  # embed_dim
                else:
                    # Fallback for unexpected shape
                    pooled = torch.nn.functional.adaptive_avg_pool2d(last_features, (1, 1))
                    return pooled.shape[1]
            else:
                # CNN models - features are [B, C, H, W]
                pooled = torch.nn.functional.adaptive_avg_pool2d(last_features, (1, 1))
                return pooled.shape[1]
    
    def _register_hooks(self):
        """Register forward hooks for feature extraction"""
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_outputs[name] = output
            return hook
        
        for level in self.feature_levels:
            try:
                # Navigate to the specified module
                module = self.model
                for attr in level.split('.'):
                    module = getattr(module, attr)
                
                handle = module.register_forward_hook(hook_fn(level))
                self.feature_hooks[level] = handle
            except AttributeError:
                print(f"Warning: Could not register hook for {level}")
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning both features and classification logits
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dict containing:
                - 'features': Feature map [B, D, H', W'] for CNN or [B, seq_len, embed_dim] for ViT
                - 'logits': Classification logits [B, num_classes]
                - 'multi_features': Dict of multi-level features (if available)
        """
        # Clear previous outputs
        self.feature_outputs.clear()
        
        # Forward pass
        features = self.model(x)
        
        # Handle different model types for pooling
        if 'vit' in self.model_name.lower() or 'deit' in self.model_name.lower():
            # Vision Transformer models - features are [B, seq_len, embed_dim]
            if features.dim() == 3:
                # Use CLS token for classification (first token)
                pooled_features = features[:, 0, :]  # [B, embed_dim]
            else:
                # Fallback to adaptive pooling if 4D
                pooled_features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                pooled_features = pooled_features.flatten(1)
        else:
            # CNN models - features are [B, C, H, W]
            pooled_features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            pooled_features = pooled_features.flatten(1)
        
        logits = self.classifier(pooled_features)
        
        # Prepare output
        output = {
            'features': features,
            'logits': logits,
            'pooled_features': pooled_features
        }
        
        # Add multi-level features if available
        if self.feature_outputs:
            output['multi_features'] = self.feature_outputs.copy()
        
        return output
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the extracted features"""
        # Get proper input size for this model
        input_size = self._get_input_size()
        dummy_input = torch.randn(1, *input_size, device=self._get_device())
        with torch.no_grad():
            output = self.forward(dummy_input)
        
        info = {
            'model_name': self.model_name,
            'input_size': input_size,
            'feature_shape': output['features'].shape[1:],
            'pooled_feature_dim': output['pooled_features'].shape[1],
            'num_classes': self.num_classes
        }
        
        if 'multi_features' in output:
            info['multi_feature_shapes'] = {
                k: v.shape[1:] for k, v in output['multi_features'].items()
            }
        
        return info


def create_timm_backbone(model_name: str = 'resnet50', 
                        pretrained: bool = True,
                        num_classes: int = 1000,
                        **kwargs) -> TimmBackbone:
    """
    Factory function to create a timm backbone
    
    Args:
        model_name: Name of the timm model
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes
        **kwargs: Additional arguments for TimmBackbone
    
    Returns:
        TimmBackbone instance
    """
    return TimmBackbone(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs
    )


def list_supported_models() -> List[str]:
    """List all supported timm models"""
    return timm.list_models()


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a specific timm model"""
    try:
        model = timm.create_model(model_name, pretrained=False)
        cfg = timm.models.get_pretrained_cfg(model_name)
        
        return {
            'model_name': model_name,
            'pretrained_cfg': cfg,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'input_size': cfg.input_size if cfg else (3, 224, 224),
            'architecture': model.__class__.__name__
        }
    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    # Test different backbones
    test_models = ['resnet50', 'efficientnet_b0', 'convnext_tiny']
    
    for model_name in test_models:
        print(f"\n=== Testing {model_name} ===")
        
        try:
            # Create backbone
            backbone = create_timm_backbone(
                model_name=model_name,
                pretrained=True,
                num_classes=80
            )
            
            # Test forward pass
            input_size = backbone._get_input_size()
            dummy_input = torch.randn(2, *input_size, device=backbone._get_device())
            output = backbone(dummy_input)
            
            print(f"Input shape: {dummy_input.shape}")
            print(f"Feature shape: {output['features'].shape}")
            print(f"Logits shape: {output['logits'].shape}")
            print(f"Pooled features shape: {output['pooled_features'].shape}")
            
            # Get feature info
            info = backbone.get_feature_info()
            print(f"Feature info: {info}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
