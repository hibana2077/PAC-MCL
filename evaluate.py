"""
Evaluation script for PAC-MCL
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import yaml
import os
import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models import PAC_MCL_Model
from src.dataset.ufgvc import UFGVCDataset
from src.utils import create_val_transforms, ModelEvaluator, ManifoldMetrics
from src.models.manifold import ManifoldDistance


def load_model_from_checkpoint(checkpoint_path: str, 
                             config: Dict[str, Any], 
                             device: torch.device) -> PAC_MCL_Model:
    """Load model from checkpoint"""
    
    # Create model
    model = PAC_MCL_Model(**config['model'])
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    return model


def evaluate_manifold_properties(model: PAC_MCL_Model,
                                dataloader: DataLoader,
                                device: torch.device,
                                output_dir: Path):
    """Evaluate manifold learning properties"""
    
    model.eval()
    
    # Collect all covariance matrices and labels
    all_covariances = []
    all_labels = []
    
    print("Extracting covariance matrices...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass to get covariance matrices
            output = model.forward_single_view(images)
            covariances = output['covariance_matrices']  # [B, P, D, D]
            
            all_covariances.append(covariances.cpu())
            all_labels.append(labels.cpu())
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
    
    # Concatenate all results
    all_covariances = torch.cat(all_covariances, dim=0)  # [N, P, D, D]
    all_labels = torch.cat(all_labels, dim=0)  # [N]
    
    print(f"Collected {all_covariances.shape[0]} samples")
    
    # Create distance calculator
    distance_calc = ManifoldDistance(metric=model.distance_metric, eps=model.eps)
    
    # Compute manifold statistics
    print("Computing manifold statistics...")
    
    intra_stats = ManifoldMetrics.compute_intra_class_distance(
        all_covariances, all_labels, distance_calc
    )
    
    inter_stats = ManifoldMetrics.compute_inter_class_distance(
        all_covariances, all_labels, distance_calc
    )
    
    margin_stats = ManifoldMetrics.compute_margin_statistics(intra_stats, inter_stats)
    
    # Print results
    print("\n" + "="*50)
    print("MANIFOLD LEARNING ANALYSIS")
    print("="*50)
    
    print(f"\nIntra-class distances ({model.distance_metric}):")
    for key, value in intra_stats.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\nInter-class distances ({model.distance_metric}):")
    for key, value in inter_stats.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\nMargin statistics:")
    for key, value in margin_stats.items():
        print(f"  {key}: {value:.6f}")
    
    # Save results
    manifold_results = {
        'intra_class': intra_stats,
        'inter_class': inter_stats,
        'margin': margin_stats,
        'distance_metric': model.distance_metric,
        'num_samples': all_covariances.shape[0],
        'num_parts': all_covariances.shape[1],
        'part_dim': all_covariances.shape[2]
    }
    
    import json
    with open(output_dir / 'manifold_analysis.json', 'w') as f:
        json.dump(manifold_results, f, indent=2)
    
    # Plot distance distributions
    plot_distance_distributions(all_covariances, all_labels, distance_calc, output_dir)
    
    return manifold_results


def plot_distance_distributions(covariances: torch.Tensor,
                               labels: torch.Tensor,
                               distance_calc,
                               output_dir: Path):
    """Plot intra vs inter class distance distributions"""
    
    print("Computing distance distributions...")
    
    # Sample subset for efficiency
    max_samples = 500
    if covariances.shape[0] > max_samples:
        indices = torch.randperm(covariances.shape[0])[:max_samples]
        covariances = covariances[indices]
        labels = labels[indices]
    
    intra_distances = []
    inter_distances = []
    
    # Compute pairwise distances
    for i in range(covariances.shape[0]):
        for j in range(i + 1, covariances.shape[0]):
            # Compute distance for first part only (for efficiency)
            dist = distance_calc(
                covariances[i, 0].unsqueeze(0),
                covariances[j, 0].unsqueeze(0)
            )[0].item()
            
            if labels[i] == labels[j]:
                intra_distances.append(dist)
            else:
                inter_distances.append(dist)
    
    # Plot distributions
    plt.figure(figsize=(10, 6))
    
    plt.hist(intra_distances, bins=50, alpha=0.7, label='Intra-class', density=True)
    plt.hist(inter_distances, bins=50, alpha=0.7, label='Inter-class', density=True)
    
    plt.xlabel('Manifold Distance')
    plt.ylabel('Density')
    plt.title('Distribution of Intra vs Inter-class Manifold Distances')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distance_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distance distribution plot saved to: {output_dir / 'distance_distributions.png'}")


def visualize_attention_maps(model: PAC_MCL_Model,
                           dataloader: DataLoader,
                           device: torch.device,
                           output_dir: Path,
                           num_samples: int = 8):
    """Visualize attention maps for sample images"""
    
    model.eval()
    
    print(f"Generating attention visualizations for {num_samples} samples...")
    
    # Get sample batch
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Select first num_samples
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    with torch.no_grad():
        output = model.forward_single_view(images)
        attention_maps = output['attention_maps']  # [B, P, H, W]
    
    # Convert images back to PIL format for visualization
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    denorm_images = images.cpu() * std + mean
    denorm_images = torch.clamp(denorm_images, 0, 1)
    
    # Create visualization
    B, P, H, W = attention_maps.shape
    
    fig, axes = plt.subplots(num_samples, P + 1, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for b in range(num_samples):
        # Original image
        axes[b, 0].imshow(denorm_images[b].permute(1, 2, 0))
        axes[b, 0].set_title(f'Original (Class {labels[b].item()})')
        axes[b, 0].axis('off')
        
        # Attention maps for each part
        for p in range(P):
            att_map = attention_maps[b, p].cpu().numpy()
            
            # Overlay attention on original image
            img_array = denorm_images[b].permute(1, 2, 0).numpy()
            
            # Resize attention map to match image size
            from skimage.transform import resize
            att_resized = resize(att_map, (224, 224))
            
            # Create overlay
            axes[b, p + 1].imshow(img_array)
            axes[b, p + 1].imshow(att_resized, alpha=0.6, cmap='hot')
            axes[b, p + 1].set_title(f'Part {p + 1}')
            axes[b, p + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Attention visualizations saved to: {output_dir / 'attention_visualizations.png'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PAC-MCL model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: same as config)')
    parser.add_argument('--manifold-analysis', action='store_true', 
                       help='Perform manifold learning analysis')
    parser.add_argument('--visualize-attention', action='store_true',
                       help='Generate attention visualizations')
    parser.add_argument('--detailed-report', action='store_true',
                       help='Generate detailed evaluation report')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config['training']['output_dir']) / 'evaluation'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset and dataloader
    print("Loading dataset...")
    
    transform = create_val_transforms(
        image_size=config['data']['image_size']
    )
    
    dataset = UFGVCDataset(
        dataset_name=config['data']['dataset_name'],
        root=config['data']['data_root'],
        split=args.split,
        transform=transform,
        download=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Evaluating on {len(dataset)} {args.split} samples")
    
    # Update config with actual number of classes
    config['model']['num_classes'] = len(dataset.classes)
    
    # Load model
    print("Loading model...")
    model = load_model_from_checkpoint(args.checkpoint, config, device)
    
    # Basic evaluation
    print("\n" + "="*50)
    print("BASIC EVALUATION")
    print("="*50)
    
    evaluator = ModelEvaluator(model, device, len(dataset.classes), dataset.classes)
    metrics = evaluator.evaluate_dataloader(dataloader, return_predictions=True)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    
    if 'top5_accuracy' in metrics:
        print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
    
    if 'ece' in metrics:
        print(f"Expected Calibration Error: {metrics['ece']:.4f}")
        print(f"Negative Log Likelihood: {metrics['nll']:.4f}")
    
    # Save basic metrics
    basic_results = {
        'split': args.split,
        'dataset': config['data']['dataset_name'],
        'model': config['model']['backbone_name'],
        'metrics': {k: v for k, v in metrics.items() if k != 'predictions'},
        'config': config
    }
    
    import json
    with open(output_dir / 'basic_evaluation.json', 'w') as f:
        # Convert numpy types
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        basic_results = convert_numpy(basic_results)
        json.dump(basic_results, f, indent=2)
    
    # Detailed report
    if args.detailed_report:
        print("\nGenerating detailed classification report...")
        report = evaluator.create_classification_report(
            dataloader, 
            save_path=output_dir / 'classification_report.txt'
        )
        
        # Confusion matrix
        evaluator.plot_confusion_matrix(
            dataloader,
            save_path=output_dir / 'confusion_matrix.png',
            normalize=True
        )
        
        # Analyze difficult classes
        difficult_analysis = evaluator.analyze_difficult_classes(dataloader, top_k=10)
        
        print("\nMost difficult classes (lowest F1):")
        for class_name, f1_score in difficult_analysis['most_difficult']:
            print(f"  {class_name}: {f1_score:.4f}")
        
        print("\nEasiest classes (highest F1):")
        for class_name, f1_score in difficult_analysis['easiest']:
            print(f"  {class_name}: {f1_score:.4f}")
    
    # Manifold analysis
    if args.manifold_analysis:
        print("\n" + "="*50)
        print("MANIFOLD ANALYSIS")
        print("="*50)
        
        manifold_results = evaluate_manifold_properties(model, dataloader, device, output_dir)
    
    # Attention visualization
    if args.visualize_attention:
        print("\n" + "="*50)
        print("ATTENTION VISUALIZATION")
        print("="*50)
        
        visualize_attention_maps(model, dataloader, device, output_dir)
    
    print(f"\nEvaluation completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
