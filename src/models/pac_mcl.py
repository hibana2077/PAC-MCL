"""
Main PAC-MCL Model
Integrates all components: backbone, part extraction, manifold operations, and losses
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import warnings

from .backbone import TimmBackbone
from .parts import PartExtractor, AdaptivePartExtractor
from .manifold import CovarianceEstimator, PartAlignment
from .losses import SymmetricPACMCL


class PAC_MCL_Model(nn.Module):
    """
    Complete PAC-MCL model combining all components
    """
    
    def __init__(self,
                 # Backbone parameters
                 backbone_name: str = 'resnet50',
                 num_classes: int = 80,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 
                 # Part extraction parameters
                 num_parts: int = 6,
                 part_dim: int = 64,
                 use_adaptive_parts: bool = False,
                 
                 # Manifold parameters
                 distance_metric: str = 'log_euclidean',
                 alignment_method: str = 'bnn',
                 shrinkage_alpha: Optional[float] = None,
                 eps: float = 1e-4,
                 
                 # Loss parameters
                 gamma: float = 0.75,
                 lambda_pos_ce: float = 1.0,
                 margin: float = 0.2):
        super().__init__()
        
        # Ensure eps is a float (handle potential string parsing issues)
        if isinstance(eps, str):
            eps = float(eps)
        
        # Store parameters
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.num_parts = num_parts
        self.part_dim = part_dim
        self.distance_metric = distance_metric
        self.alignment_method = alignment_method
        
        # Create backbone
        self.backbone = TimmBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone
        )
        
        # Get feature dimension from backbone
        feature_info = self.backbone.get_feature_info()
        self.backbone_feature_dim = feature_info['feature_shape'][0]  # Channel dimension
        
        # Part extraction
        if use_adaptive_parts:
            self.part_extractor = AdaptivePartExtractor(
                in_channels=self.backbone_feature_dim,
                base_num_parts=num_parts,
                part_dim=part_dim,
                adaptive_parts=True
            )
        else:
            self.part_extractor = PartExtractor(
                in_channels=self.backbone_feature_dim,
                num_parts=num_parts,
                part_dim=part_dim,
                use_attention=True
            )
        
        # Covariance estimation
        self.covariance_estimator = CovarianceEstimator(
            shrinkage_alpha=shrinkage_alpha,
            eps=eps,
            center_data=True
        )
        
        # Part alignment
        self.part_aligner = PartAlignment(
            alignment_method=alignment_method,
            distance_metric=distance_metric
        )
        
        # Loss function
        self.loss_fn = SymmetricPACMCL(
            gamma=gamma,
            lambda_pos_ce=lambda_pos_ce,
            distance_metric=distance_metric,
            margin=margin
        )
        
        # Training state
        self.training_step = 0
        self.current_epoch = 0
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from input images
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Dict containing backbone outputs
        """
        return self.backbone(x)
    
    def extract_parts(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract parts from feature maps
        
        Args:
            features: Feature maps [B, C, H, W]
            
        Returns:
            Dict containing part features and attention maps
        """
        return self.part_extractor(features)
    
    def compute_covariance_matrices(self, part_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute covariance matrices for parts using spatial features
        
        Args:
            part_output: Output from part extractor containing spatial features and attention maps
            
        Returns:
            Covariance matrices [B, P, part_dim, part_dim]
        """
        spatial_features = part_output['spatial_features']  # [B, part_dim, H, W]
        attention_maps = part_output['attention_maps']  # [B, P, H, W]
        
        B, part_dim, H, W = spatial_features.shape
        _, P, _, _ = attention_maps.shape
        
        covariance_matrices = []
        
        for b in range(B):
            batch_covs = []
            for p in range(P):
                # Get spatial features and attention for this batch and part
                spatial_feat = spatial_features[b]  # [part_dim, H, W]
                attention = attention_maps[b, p]  # [H, W]
                
                # Flatten spatial features
                spatial_flat = spatial_feat.view(part_dim, -1).t()  # [H*W, part_dim]
                attention_flat = attention.view(-1)  # [H*W]
                
                # Weight spatial features by attention
                attention_weights = attention_flat / (attention_flat.sum() + 1e-8)
                weighted_features = spatial_flat * attention_weights.unsqueeze(1)
                
                # Sample features based on attention (top-k approach)
                num_samples = min(50, H * W)  # Sample at most 50 points
                _, top_indices = torch.topk(attention_flat, num_samples)
                sampled_features = spatial_flat[top_indices]  # [num_samples, part_dim]
                
                # Compute covariance
                cov = self.covariance_estimator(sampled_features.unsqueeze(0))  # [1, part_dim, part_dim]
                batch_covs.append(cov[0])
            
            covariance_matrices.append(torch.stack(batch_covs))  # [P, part_dim, part_dim]
        
        return torch.stack(covariance_matrices)  # [B, P, part_dim, part_dim]
    
    def forward_single_view(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a single view
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Dict containing logits, features, parts, and covariance matrices
        """
        # Extract backbone features
        backbone_output = self.extract_features(x)
        features = backbone_output['features']
        logits = backbone_output['logits']
        
        # Extract parts
        part_output = self.extract_parts(features)
        
        # Compute covariance matrices
        covariance_matrices = self.compute_covariance_matrices(part_output)
        
        return {
            'logits': logits,
            'features': features,
            'part_features': part_output['part_features'],
            'attention_maps': part_output['attention_maps'],
            'covariance_matrices': covariance_matrices
        }
    
    def forward(self, 
                x1: torch.Tensor, 
                x2: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_loss: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass for dual views (training mode)
        
        Args:
            x1, x2: Two augmented views of the same images [B, C, H, W]
            labels: Ground truth labels [B] (required if return_loss=True)
            return_loss: Whether to compute and return loss
            
        Returns:
            Dict containing outputs and optionally loss
        """
        # Forward pass for both views
        output_v1 = self.forward_single_view(x1)
        output_v2 = self.forward_single_view(x2)
        
        result = {
            'logits_v1': output_v1['logits'],
            'logits_v2': output_v2['logits'],
            'features_v1': output_v1['features'],
            'features_v2': output_v2['features'],
            'attention_maps_v1': output_v1['attention_maps'],
            'attention_maps_v2': output_v2['attention_maps'],
            'covariance_matrices_v1': output_v1['covariance_matrices'],
            'covariance_matrices_v2': output_v2['covariance_matrices']
        }
        
        if return_loss and labels is not None:
            # Compute part alignments
            alignments, distance_matrices = self.part_aligner(
                output_v1['covariance_matrices'],
                output_v2['covariance_matrices']
            )
            
            # Compute loss
            loss, loss_dict = self.loss_fn(
                output_v1['logits'],
                output_v2['logits'],
                labels,
                output_v1['covariance_matrices'],
                output_v2['covariance_matrices'],
                alignments
            )
            
            result.update({
                'loss': loss,
                'loss_dict': loss_dict,
                'alignments': alignments,
                'distance_matrices': distance_matrices
            })
        
        return result
    
    def forward_inference(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for inference (single view)
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Dict containing inference outputs
        """
        with torch.no_grad():
            output = self.forward_single_view(x)
            
            # For inference, we mainly need logits and features
            return {
                'logits': output['logits'],
                'features': output['features'],
                'attention_maps': output['attention_maps'],
                'predictions': torch.softmax(output['logits'], dim=1)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        backbone_info = self.backbone.get_feature_info()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone_name': self.backbone_name,
            'num_classes': self.num_classes,
            'num_parts': self.num_parts,
            'part_dim': self.part_dim,
            'distance_metric': self.distance_metric,
            'alignment_method': self.alignment_method,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_info': backbone_info
        }
    
    def set_epoch(self, epoch: int):
        """Set current epoch for potential curriculum learning"""
        self.current_epoch = epoch
        
        # Example: Switch from Log-Euclidean to Bures-Wasserstein after certain epochs
        if hasattr(self, 'switch_epoch') and epoch >= self.switch_epoch:
            if self.distance_metric == 'log_euclidean':
                self.distance_metric = 'bures_wasserstein'
                # Update distance metric in components
                if hasattr(self.part_aligner.distance_calculator, 'metric'):
                    self.part_aligner.distance_calculator.metric = 'bures_wasserstein'
                if hasattr(self.loss_fn.total_loss.pac_mcl_loss.distance_calculator, 'metric'):
                    self.loss_fn.total_loss.pac_mcl_loss.distance_calculator.metric = 'bures_wasserstein'


def create_pac_mcl_model(config: Dict[str, Any]) -> PAC_MCL_Model:
    """
    Factory function to create PAC-MCL model from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PAC_MCL_Model instance
    """
    return PAC_MCL_Model(**config)


if __name__ == "__main__":
    # Test the complete model
    print("Testing PAC-MCL model...")
    
    # Model configuration
    config = {
        'backbone_name': 'resnet50',
        'num_classes': 80,
        'pretrained': True,
        'num_parts': 6,
        'part_dim': 64,
        'distance_metric': 'log_euclidean',
        'gamma': 0.75,
        'margin': 0.2
    }
    
    # Create model
    model = create_pac_mcl_model(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test data
    batch_size = 2
    x1 = torch.randn(batch_size, 3, 224, 224)
    x2 = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 80, (batch_size,))
    
    print(f"Input shapes: x1={x1.shape}, x2={x2.shape}, labels={labels.shape}")
    
    # Test training forward pass
    print("\n=== Testing training forward pass ===")
    model.train()
    
    try:
        output = model(x1, x2, labels, return_loss=True)
        
        print("Training outputs:")
        print(f"  Logits v1: {output['logits_v1'].shape}")
        print(f"  Logits v2: {output['logits_v2'].shape}")
        print(f"  Covariance matrices v1: {output['covariance_matrices_v1'].shape}")
        print(f"  Covariance matrices v2: {output['covariance_matrices_v2'].shape}")
        print(f"  Loss: {output['loss'].item():.4f}")
        
        print("  Loss components:")
        for key, value in output['loss_dict'].items():
            if hasattr(value, 'item'):
                print(f"    {key}: {value.item():.4f}")
        
    except Exception as e:
        print(f"Training forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test inference forward pass
    print("\n=== Testing inference forward pass ===")
    model.eval()
    
    try:
        with torch.no_grad():
            inference_output = model.forward_inference(x1)
        
        print("Inference outputs:")
        print(f"  Logits: {inference_output['logits'].shape}")
        print(f"  Predictions: {inference_output['predictions'].shape}")
        print(f"  Features: {inference_output['features'].shape}")
        print(f"  Attention maps: {inference_output['attention_maps'].shape}")
        
    except Exception as e:
        print(f"Inference forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test model info
    print("\n=== Model Information ===")
    info = model.get_model_info()
    for key, value in info.items():
        if key != 'backbone_info':
            print(f"{key}: {value}")
    
    print("\nModel test completed!")
