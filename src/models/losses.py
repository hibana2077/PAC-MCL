"""
PAC-MCL Loss Functions
Implements Part-Aware Manifold Contrastive Loss and related components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import random


class HardNegativeMiner:
    """
    Mine hard negative samples for contrastive learning
    """
    
    @staticmethod
    def mine_hard_negatives(anchor_matrices: torch.Tensor,
                          negative_matrices: List[torch.Tensor],
                          distance_calculator,
                          topk: int = 5,
                          strategy: str = 'hardest') -> List[List[Tuple[float, int, int]]]:
        """
        Mine hard negatives for each anchor part
        
        Args:
            anchor_matrices: Anchor covariance matrices [P, D, D]
            negative_matrices: List of negative covariance matrices [[P, D, D], ...]
            distance_calculator: ManifoldDistance instance
            topk: Number of hard negatives to select
            strategy: 'hardest', 'random', or 'semi_hard'
            
        Returns:
            List of hard negatives for each part: [[distance, neg_idx, part_idx], ...]
        """
        P = anchor_matrices.shape[0]
        hard_negatives = []
        
        for p in range(P):
            anchor_part = anchor_matrices[p].unsqueeze(0)  # [1, D, D]
            candidates = []
            
            # Collect all negative candidates
            for neg_idx, neg_matrices in enumerate(negative_matrices):
                for q in range(neg_matrices.shape[0]):
                    neg_part = neg_matrices[q].unsqueeze(0)  # [1, D, D]
                    
                    # Compute distance
                    distance = distance_calculator(anchor_part, neg_part)[0].item()
                    candidates.append((distance, neg_idx, q))
            
            # Select hard negatives based on strategy
            if strategy == 'hardest':
                # Sort by distance (ascending - closer is harder)
                candidates.sort(key=lambda x: x[0])
                hard_for_part = candidates[:topk]
            elif strategy == 'random':
                # Random selection
                hard_for_part = random.sample(candidates, min(topk, len(candidates)))
            elif strategy == 'semi_hard':
                # Select negatives that are closer than average but not too close
                distances = [c[0] for c in candidates]
                mean_dist = sum(distances) / len(distances)
                semi_hard = [c for c in candidates if c[0] < mean_dist]
                semi_hard.sort(key=lambda x: x[0])
                hard_for_part = semi_hard[:topk] if semi_hard else candidates[:topk]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            hard_negatives.append(hard_for_part)
        
        return hard_negatives


class PAC_MCL_Loss(nn.Module):
    """
    Part-Aware Manifold Contrastive Loss
    Implements triplet-style loss on the SPD manifold
    """
    
    def __init__(self,
                 distance_metric: str = 'log_euclidean',
                 margin: float = 0.2,
                 reduction_method: str = 'min',
                 temperature: float = 0.05,
                 eps: float = 1e-4):
        super().__init__()
        
        from .manifold import ManifoldDistance
        
        self.distance_calculator = ManifoldDistance(metric=distance_metric, eps=eps)
        self.margin = margin
        self.reduction_method = reduction_method  # 'min', 'mean', 'softmin'
        self.temperature = temperature
        
        self.miner = HardNegativeMiner()
    
    def compute_positive_distance(self,
                                anchor_matrices: torch.Tensor,
                                positive_matrices: torch.Tensor,
                                alignment: Dict[int, int]) -> torch.Tensor:
        """
        Compute positive pair distances using alignment
        
        Args:
            anchor_matrices: [P, D, D]
            positive_matrices: [P, D, D] 
            alignment: Part alignment mapping
            
        Returns:
            Positive distances [P]
        """
        P = anchor_matrices.shape[0]
        pos_distances = torch.zeros(P, device=anchor_matrices.device)
        
        for p in range(P):
            if p in alignment:
                q = alignment[p]
                anchor_part = anchor_matrices[p].unsqueeze(0)
                pos_part = positive_matrices[q].unsqueeze(0)
                pos_distances[p] = self.distance_calculator(anchor_part, pos_part)[0]
            else:
                # If no alignment, use identity mapping
                anchor_part = anchor_matrices[p].unsqueeze(0)
                pos_part = positive_matrices[p].unsqueeze(0)
                pos_distances[p] = self.distance_calculator(anchor_part, pos_part)[0]
        
        return pos_distances
    
    def compute_negative_distance(self,
                                hard_negatives: List[List[Tuple[float, int, int]]]) -> torch.Tensor:
        """
        Compute negative distances using hard negative mining results
        
        Args:
            hard_negatives: Hard negatives for each part
            
        Returns:
            Negative distances [P]
        """
        P = len(hard_negatives)
        neg_distances = torch.zeros(P)
        
        for p in range(P):
            hard_list = hard_negatives[p]
            
            if not hard_list:
                neg_distances[p] = float('inf')  # No negatives available
                continue
            
            distances = [h[0] for h in hard_list]
            
            if self.reduction_method == 'min':
                neg_distances[p] = min(distances)
            elif self.reduction_method == 'mean':
                neg_distances[p] = sum(distances) / len(distances)
            elif self.reduction_method == 'softmin':
                # Temperature-scaled soft minimum
                weights = F.softmax(torch.tensor([-d/self.temperature for d in distances]), dim=0)
                neg_distances[p] = sum(w * d for w, d in zip(weights, distances))
            else:
                neg_distances[p] = min(distances)
        
        return neg_distances
    
    def forward(self,
                anchor_matrices: torch.Tensor,
                positive_matrices: torch.Tensor,
                negative_matrices: List[torch.Tensor],
                alignment: Dict[int, int],
                topk_negatives: int = 5) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute PAC-MCL loss
        
        Args:
            anchor_matrices: Anchor covariance matrices [P, D, D]
            positive_matrices: Positive covariance matrices [P, D, D]
            negative_matrices: List of negative covariance matrices
            alignment: Part alignment mapping
            topk_negatives: Number of hard negatives to consider
            
        Returns:
            Tuple of (loss, info_dict)
        """
        # Mine hard negatives
        hard_negatives = self.miner.mine_hard_negatives(
            anchor_matrices, negative_matrices, 
            self.distance_calculator, topk_negatives
        )
        
        # Compute positive distances
        pos_distances = self.compute_positive_distance(
            anchor_matrices, positive_matrices, alignment
        )
        
        # Compute negative distances
        neg_distances = self.compute_negative_distance(hard_negatives)
        neg_distances = neg_distances.to(pos_distances.device)
        
        # Compute triplet loss
        triplet_losses = F.relu(pos_distances - neg_distances + self.margin)
        
        # Average over parts
        loss = torch.mean(triplet_losses)
        
        # Prepare info dictionary
        info = {
            'pos_distances': pos_distances,
            'neg_distances': neg_distances,
            'triplet_losses': triplet_losses,
            'num_active_triplets': torch.sum(triplet_losses > 0).float(),
            'margin_violations': torch.sum(pos_distances > neg_distances).float(),
        }
        
        return loss, info


class TotalLoss(nn.Module):
    """
    Total loss combining classification and PAC-MCL
    """
    
    def __init__(self,
                 gamma: float = 0.75,
                 lambda_pos_ce: float = 1.0,
                 distance_metric: str = 'log_euclidean',
                 margin: float = 0.2):
        super().__init__()
        
        self.gamma = gamma
        self.lambda_pos_ce = lambda_pos_ce
        
        # Loss components
        self.ce_loss = nn.CrossEntropyLoss()
        self.pac_mcl_loss = PAC_MCL_Loss(
            distance_metric=distance_metric,
            margin=margin
        )
    
    def forward(self,
                logits_main: torch.Tensor,
                logits_pos: torch.Tensor,
                labels: torch.Tensor,
                anchor_matrices: torch.Tensor,
                positive_matrices: torch.Tensor,
                negative_matrices_list: List[torch.Tensor],
                alignment: Dict[int, int]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss
        
        Args:
            logits_main: Main view logits [B, num_classes]
            logits_pos: Positive view logits [B, num_classes]
            labels: Ground truth labels [B]
            anchor_matrices: Anchor covariance matrices [B, P, D, D]
            positive_matrices: Positive covariance matrices [B, P, D, D]
            negative_matrices_list: List of negative matrices for each sample
            alignment: Part alignment mapping
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        batch_size = logits_main.shape[0]
        
        # Classification losses
        ce_main = self.ce_loss(logits_main, labels)
        ce_pos = self.ce_loss(logits_pos, labels)
        
        # PAC-MCL loss (computed per sample and averaged)
        pac_mcl_losses = []
        pac_mcl_infos = []
        
        for b in range(batch_size):
            # Get negatives for this sample (exclude itself)
            negatives = [negative_matrices_list[i][b] for i in range(len(negative_matrices_list)) if i != b]
            
            if negatives:  # Only compute if negatives are available
                pac_loss, pac_info = self.pac_mcl_loss(
                    anchor_matrices[b],
                    positive_matrices[b], 
                    negatives,
                    alignment
                )
                pac_mcl_losses.append(pac_loss)
                pac_mcl_infos.append(pac_info)
        
        # Average PAC-MCL loss
        if pac_mcl_losses:
            pac_mcl_total = torch.stack(pac_mcl_losses).mean()
        else:
            pac_mcl_total = torch.tensor(0.0, device=logits_main.device)
        
        # Total loss
        total_loss = ce_main + self.lambda_pos_ce * ce_pos + self.gamma * pac_mcl_total
        
        # Prepare loss dictionary
        loss_dict = {
            'ce_main': ce_main,
            'ce_pos': ce_pos,
            'pac_mcl': pac_mcl_total,
            'total': total_loss
        }
        
        # Add PAC-MCL info if available
        if pac_mcl_infos:
            avg_pos_dist = torch.stack([info['pos_distances'].mean() for info in pac_mcl_infos]).mean()
            avg_neg_dist = torch.stack([info['neg_distances'].mean() for info in pac_mcl_infos]).mean()
            avg_active_triplets = torch.stack([info['num_active_triplets'] for info in pac_mcl_infos]).mean()
            
            loss_dict.update({
                'avg_pos_distance': avg_pos_dist,
                'avg_neg_distance': avg_neg_dist,
                'avg_active_triplets': avg_active_triplets
            })
        
        return total_loss, loss_dict


class SymmetricPACMCL(nn.Module):
    """
    Symmetric PAC-MCL that treats both views as anchors
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.total_loss = TotalLoss(**kwargs)
    
    def forward(self,
                logits_v1: torch.Tensor,
                logits_v2: torch.Tensor,
                labels: torch.Tensor,
                matrices_v1: torch.Tensor,
                matrices_v2: torch.Tensor,
                alignments: List[Dict[int, int]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute symmetric PAC-MCL loss
        
        Args:
            logits_v1, logits_v2: Logits from both views
            labels: Ground truth labels
            matrices_v1, matrices_v2: Covariance matrices from both views
            alignments: List of alignment mappings for each sample
            
        Returns:
            Tuple of (loss, loss_dict)
        """
        batch_size = matrices_v1.shape[0]
        
        # Prepare negative matrices
        negatives_v1 = [matrices_v2[i] for i in range(batch_size)]
        negatives_v2 = [matrices_v1[i] for i in range(batch_size)]
        
        # Compute loss for v1 as anchor
        loss_1, dict_1 = self.total_loss(
            logits_v1, logits_v2, labels,
            matrices_v1, matrices_v2, negatives_v1,
            alignments[0] if alignments else {}
        )
        
        # Compute loss for v2 as anchor  
        loss_2, dict_2 = self.total_loss(
            logits_v2, logits_v1, labels,
            matrices_v2, matrices_v1, negatives_v2,
            alignments[0] if alignments else {}  # Use same alignment
        )
        
        # Average the losses
        total_loss = 0.5 * (loss_1 + loss_2)
        
        # Combine loss dictionaries
        combined_dict = {}
        for key in dict_1.keys():
            if key != 'total':
                combined_dict[f'{key}_v1'] = dict_1[key]
                combined_dict[f'{key}_v2'] = dict_2[key]
                combined_dict[key] = 0.5 * (dict_1[key] + dict_2[key])
        
        combined_dict['total'] = total_loss
        
        return total_loss, combined_dict


if __name__ == "__main__":
    # Test loss functions
    print("Testing PAC-MCL loss functions...")
    
    # Test parameters
    batch_size = 2
    num_classes = 80
    num_parts = 6
    feature_dim = 64
    
    # Create dummy data
    logits_v1 = torch.randn(batch_size, num_classes)
    logits_v2 = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Create dummy covariance matrices
    def create_dummy_cov_matrices(batch_size, num_parts, dim):
        matrices = []
        for _ in range(batch_size):
            batch_matrices = []
            for _ in range(num_parts):
                A = torch.randn(dim, dim)
                cov = torch.mm(A, A.t()) + 0.1 * torch.eye(dim)
                batch_matrices.append(cov)
            matrices.append(torch.stack(batch_matrices))
        return torch.stack(matrices)
    
    matrices_v1 = create_dummy_cov_matrices(batch_size, num_parts, feature_dim)
    matrices_v2 = create_dummy_cov_matrices(batch_size, num_parts, feature_dim)
    
    print(f"Matrices v1 shape: {matrices_v1.shape}")
    print(f"Matrices v2 shape: {matrices_v2.shape}")
    
    # Test PAC-MCL Loss
    print("\n=== Testing PAC-MCL Loss ===")
    pac_mcl = PAC_MCL_Loss(distance_metric='log_euclidean', margin=0.2)
    
    # Create dummy alignment
    alignment = {i: i for i in range(num_parts)}  # Identity alignment
    negatives = [matrices_v2[1]]  # Use second sample as negative
    
    loss, info = pac_mcl(matrices_v1[0], matrices_v2[0], negatives, alignment)
    print(f"PAC-MCL loss: {loss.item():.4f}")
    print(f"Positive distances: {info['pos_distances']}")
    print(f"Active triplets: {info['num_active_triplets'].item()}")
    
    # Test Total Loss
    print("\n=== Testing Total Loss ===")
    total_loss_fn = TotalLoss(gamma=0.75, lambda_pos_ce=1.0)
    
    # Prepare negatives list
    negatives_list = [matrices_v2]  # Simple case
    
    total_loss, loss_dict = total_loss_fn(
        logits_v1[0:1], logits_v2[0:1], labels[0:1],
        matrices_v1[0:1], matrices_v2[0:1], negatives_list, alignment
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print("Loss components:")
    for key, value in loss_dict.items():
        if hasattr(value, 'item'):
            print(f"  {key}: {value.item():.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test Symmetric PAC-MCL
    print("\n=== Testing Symmetric PAC-MCL ===")
    symmetric_loss = SymmetricPACMCL(gamma=0.75, lambda_pos_ce=1.0)
    
    alignments = [alignment]  # List of alignments for each sample
    
    sym_loss, sym_dict = symmetric_loss(
        logits_v1, logits_v2, labels,
        matrices_v1, matrices_v2, alignments
    )
    
    print(f"Symmetric loss: {sym_loss.item():.4f}")
    print("Symmetric loss components:")
    for key, value in sym_dict.items():
        if hasattr(value, 'item'):
            print(f"  {key}: {value.item():.4f}")
    
    print("\nAll tests passed!")
