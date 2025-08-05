"""
Metrics calculation and model evaluation utilities
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd


class MetricsCalculator:
    """
    Calculate various metrics for classification tasks
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        
        # Storage for batch-wise metrics
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.losses = []
    
    def update(self, 
               predictions: torch.Tensor,
               targets: torch.Tensor,
               probabilities: Optional[torch.Tensor] = None,
               loss: Optional[float] = None):
        """
        Update metrics with batch results
        
        Args:
            predictions: Predicted class indices [B]
            targets: True class indices [B]
            probabilities: Class probabilities [B, num_classes] (optional)
            loss: Batch loss (optional)
        """
        # Convert to numpy
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        self.predictions.extend(pred_np)
        self.targets.extend(target_np)
        
        if probabilities is not None:
            prob_np = probabilities.detach().cpu().numpy()
            self.probabilities.extend(prob_np)
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute_basic_metrics(self) -> Dict[str, float]:
        """Compute basic classification metrics"""
        if not self.predictions:
            return {}
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
        
        return metrics
    
    def compute_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics"""
        if not self.predictions:
            return {}
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        # Per-class precision, recall, f1
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names[:len(precision_per_class)]):
            per_class_metrics[class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }
        
        return per_class_metrics
    
    def compute_top_k_accuracy(self, k: int = 5) -> float:
        """Compute top-k accuracy"""
        if not self.probabilities:
            return 0.0
        
        y_true = np.array(self.targets)
        y_prob = np.array(self.probabilities)
        
        # Get top-k predictions
        top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
        
        # Check if true label is in top-k
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def compute_calibration_metrics(self, n_bins: int = 10) -> Dict[str, float]:
        """Compute calibration metrics (ECE, NLL)"""
        if not self.probabilities:
            return {}
        
        y_true = np.array(self.targets)
        y_prob = np.array(self.probabilities)
        
        # Get predicted probabilities for true classes
        true_class_probs = y_prob[range(len(y_true)), y_true]
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (true_class_probs > bin_lower) & (true_class_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (np.array(self.predictions)[in_bin] == y_true[in_bin]).mean()
                avg_confidence_in_bin = true_class_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Negative Log Likelihood
        nll = -np.mean(np.log(true_class_probs + 1e-8))
        
        return {
            'ece': float(ece),
            'nll': float(nll)
        }
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        if not self.predictions:
            return np.array([])
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        return confusion_matrix(y_true, y_pred)
    
    def compute_all_metrics(self) -> Dict[str, Any]:
        """Compute all available metrics"""
        metrics = {}
        
        # Basic metrics
        basic = self.compute_basic_metrics()
        metrics.update(basic)
        
        # Per-class metrics
        per_class = self.compute_per_class_metrics()
        metrics['per_class'] = per_class
        
        # Top-k accuracy
        if self.probabilities:
            metrics['top5_accuracy'] = self.compute_top_k_accuracy(k=5)
            
            # Calibration metrics
            calibration = self.compute_calibration_metrics()
            metrics.update(calibration)
        
        # Confusion matrix
        cm = self.get_confusion_matrix()
        if cm.size > 0:
            metrics['confusion_matrix'] = cm
        
        return metrics


class ModelEvaluator:
    """
    Comprehensive model evaluation on validation/test sets
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 num_classes: int,
                 class_names: Optional[List[str]] = None):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        
        self.metrics_calc = MetricsCalculator(num_classes, class_names)
    
    def evaluate_dataloader(self, 
                           dataloader,
                           return_predictions: bool = False) -> Dict[str, Any]:
        """
        Evaluate model on a dataloader
        
        Args:
            dataloader: DataLoader to evaluate on
            return_predictions: Whether to return individual predictions
            
        Returns:
            Dictionary containing all metrics and optionally predictions
        """
        self.model.eval()
        self.metrics_calc.reset()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward_inference'):
                    # Use inference method if available
                    outputs = self.model.forward_inference(images)
                    logits = outputs['logits']
                    probabilities = outputs['predictions']
                else:
                    # Standard forward pass
                    logits = self.model(images)
                    probabilities = torch.softmax(logits, dim=1)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Update metrics
                loss = nn.CrossEntropyLoss()(logits, targets).item()
                self.metrics_calc.update(predictions, targets, probabilities, loss)
                
                # Store for return if requested
                if return_predictions:
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
        
        # Compute all metrics
        metrics = self.metrics_calc.compute_all_metrics()
        
        # Add predictions if requested
        if return_predictions:
            metrics['predictions'] = {
                'predicted': all_predictions,
                'targets': all_targets,
                'probabilities': all_probabilities
            }
        
        return metrics
    
    def create_classification_report(self, 
                                   dataloader,
                                   save_path: Optional[str] = None) -> str:
        """Create detailed classification report"""
        metrics = self.evaluate_dataloader(dataloader, return_predictions=True)
        
        y_true = metrics['predictions']['targets']
        y_pred = metrics['predictions']['predicted']
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_confusion_matrix(self, 
                            dataloader,
                            save_path: Optional[str] = None,
                            normalize: bool = True) -> plt.Figure:
        """Plot confusion matrix"""
        metrics = self.evaluate_dataloader(dataloader)
        cm = metrics['confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, 
                    annot=True, 
                    fmt=fmt,
                    cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def analyze_difficult_classes(self, 
                                dataloader,
                                top_k: int = 10) -> Dict[str, Any]:
        """Analyze most difficult classes"""
        per_class_metrics = self.evaluate_dataloader(dataloader)['per_class']
        
        # Sort by F1 score
        class_f1_scores = [(class_name, metrics['f1']) 
                          for class_name, metrics in per_class_metrics.items()]
        class_f1_scores.sort(key=lambda x: x[1])
        
        # Most difficult classes (lowest F1)
        most_difficult = class_f1_scores[:top_k]
        
        # Easiest classes (highest F1)
        easiest = class_f1_scores[-top_k:][::-1]
        
        return {
            'most_difficult': most_difficult,
            'easiest': easiest,
            'all_scores': class_f1_scores
        }


class ManifoldMetrics:
    """
    Metrics specific to manifold learning and part-aware methods
    """
    
    @staticmethod
    def compute_intra_class_distance(covariance_matrices: torch.Tensor,
                                   labels: torch.Tensor,
                                   distance_fn) -> Dict[str, float]:
        """
        Compute intra-class manifold distances
        
        Args:
            covariance_matrices: [B, P, D, D] covariance matrices
            labels: [B] class labels
            distance_fn: Function to compute manifold distance
            
        Returns:
            Dictionary with intra-class distance statistics
        """
        unique_labels = torch.unique(labels)
        intra_distances = []
        
        for label in unique_labels:
            # Get matrices for this class
            class_mask = labels == label
            class_matrices = covariance_matrices[class_mask]
            
            if class_matrices.shape[0] < 2:
                continue
            
            # Compute pairwise distances within class
            class_distances = []
            for i in range(class_matrices.shape[0]):
                for j in range(i + 1, class_matrices.shape[0]):
                    for p in range(class_matrices.shape[1]):  # For each part
                        dist = distance_fn(
                            class_matrices[i, p].unsqueeze(0),
                            class_matrices[j, p].unsqueeze(0)
                        )[0].item()
                        class_distances.append(dist)
            
            intra_distances.extend(class_distances)
        
        if not intra_distances:
            return {'mean': 0.0, 'std': 0.0, 'median': 0.0}
        
        intra_distances = np.array(intra_distances)
        return {
            'mean': float(np.mean(intra_distances)),
            'std': float(np.std(intra_distances)),
            'median': float(np.median(intra_distances)),
            'min': float(np.min(intra_distances)),
            'max': float(np.max(intra_distances))
        }
    
    @staticmethod
    def compute_inter_class_distance(covariance_matrices: torch.Tensor,
                                   labels: torch.Tensor,
                                   distance_fn) -> Dict[str, float]:
        """Compute inter-class manifold distances"""
        unique_labels = torch.unique(labels)
        inter_distances = []
        
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if i >= j:
                    continue
                
                # Get matrices for both classes
                mask1 = labels == label1
                mask2 = labels == label2
                
                matrices1 = covariance_matrices[mask1]
                matrices2 = covariance_matrices[mask2]
                
                # Compute cross-class distances
                for m1 in matrices1:
                    for m2 in matrices2:
                        for p in range(m1.shape[0]):  # For each part
                            dist = distance_fn(
                                m1[p].unsqueeze(0),
                                m2[p].unsqueeze(0)
                            )[0].item()
                            inter_distances.append(dist)
        
        if not inter_distances:
            return {'mean': 0.0, 'std': 0.0, 'median': 0.0}
        
        inter_distances = np.array(inter_distances)
        return {
            'mean': float(np.mean(inter_distances)),
            'std': float(np.std(inter_distances)),
            'median': float(np.median(inter_distances)),
            'min': float(np.min(inter_distances)),
            'max': float(np.max(inter_distances))
        }
    
    @staticmethod
    def compute_margin_statistics(intra_stats: Dict[str, float],
                                inter_stats: Dict[str, float]) -> Dict[str, float]:
        """Compute margin statistics (inter - intra distances)"""
        return {
            'margin_mean': inter_stats['mean'] - intra_stats['mean'],
            'margin_median': inter_stats['median'] - intra_stats['median'],
            'separation_ratio': inter_stats['mean'] / (intra_stats['mean'] + 1e-8)
        }


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing metrics calculation...")
    
    # Create dummy data
    num_classes = 10
    num_samples = 100
    
    # Simulate predictions and targets
    targets = torch.randint(0, num_classes, (num_samples,))
    # Make predictions somewhat correlated with targets for realistic metrics
    predictions = targets.clone()
    # Add some noise
    noise_mask = torch.rand(num_samples) < 0.2  # 20% wrong predictions
    predictions[noise_mask] = torch.randint(0, num_classes, (noise_mask.sum(),))
    
    # Generate probabilities
    probabilities = torch.softmax(torch.randn(num_samples, num_classes), dim=1)
    
    print(f"Generated {num_samples} samples with {num_classes} classes")
    
    # Test MetricsCalculator
    print("\n=== Testing MetricsCalculator ===")
    calc = MetricsCalculator(num_classes)
    
    # Simulate batch updates
    batch_size = 20
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        calc.update(
            predictions[i:end_idx],
            targets[i:end_idx],
            probabilities[i:end_idx],
            loss=0.5 + 0.3 * torch.rand(1).item()
        )
    
    # Compute metrics
    metrics = calc.compute_all_metrics()
    
    print("Basic metrics:")
    for key, value in metrics.items():
        if key not in ['per_class', 'confusion_matrix']:
            print(f"  {key}: {value:.4f}")
    
    print(f"Per-class metrics computed for {len(metrics['per_class'])} classes")
    print(f"Confusion matrix shape: {metrics['confusion_matrix'].shape}")
    
    # Test manifold metrics
    print("\n=== Testing Manifold Metrics ===")
    
    # Create dummy covariance matrices
    batch_size = 20
    num_parts = 6
    feature_dim = 32
    
    # Create random SPD matrices
    def create_spd_matrices(B, P, D):
        matrices = []
        for _ in range(B):
            part_matrices = []
            for _ in range(P):
                A = torch.randn(D, D)
                spd = torch.mm(A, A.t()) + 0.1 * torch.eye(D)
                part_matrices.append(spd)
            matrices.append(torch.stack(part_matrices))
        return torch.stack(matrices)
    
    cov_matrices = create_spd_matrices(batch_size, num_parts, feature_dim)
    labels = torch.randint(0, 5, (batch_size,))  # 5 classes
    
    # Simple distance function for testing
    def simple_distance(A, B):
        return torch.norm(A - B, p='fro').unsqueeze(0)
    
    intra_stats = ManifoldMetrics.compute_intra_class_distance(
        cov_matrices, labels, simple_distance
    )
    inter_stats = ManifoldMetrics.compute_inter_class_distance(
        cov_matrices, labels, simple_distance
    )
    margin_stats = ManifoldMetrics.compute_margin_statistics(intra_stats, inter_stats)
    
    print("Intra-class distance stats:")
    for key, value in intra_stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("Inter-class distance stats:")
    for key, value in inter_stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("Margin stats:")
    for key, value in margin_stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nAll metrics tests passed!")
