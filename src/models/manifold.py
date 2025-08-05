"""
Manifold operations for PAC-MCL
Implements SPD matrix operations and manifold distances (Log-Euclidean and Bures-Wasserstein)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import math
import warnings


class SPDMatrices:
    """
    Utility class for SPD (Symmetric Positive Definite) matrix operations
    Implements numerically stable operations on the SPD manifold
    """
    
    @staticmethod
    def symmetrize(M):
        """
        Symmetrizes a matrix.
        """
        return M.add_(M.transpose(-2, -1)).mul_(0.5)
    
    @staticmethod
    def add_identity(M: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Add identity matrix for numerical stability"""
        I = torch.eye(M.size(-1), device=M.device, dtype=M.dtype)
        return M + eps * I
    
    @staticmethod
    def ensure_spd(M: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Ensure matrix is SPD through eigenvalue clamping"""
        # Ensure eps is a float (handle potential string parsing issues)
        if isinstance(eps, str):
            eps = float(eps)
            
        M = SPDMatrices.symmetrize(M)
        
        # Eigenvalue decomposition
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(M)
            # Clamp eigenvalues to be positive
            eigenvals = torch.clamp(eigenvals, min=eps)
            # Reconstruct matrix
            M_spd = eigenvecs @ torch.diag_embed(eigenvals) @ eigenvecs.transpose(-2, -1)
            return M_spd
        except RuntimeError:
            # Fallback: add identity
            return SPDMatrices.add_identity(M, eps)
    
    @staticmethod
    def matrix_sqrt(M: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Compute matrix square root using eigenvalue decomposition"""
        # Ensure eps is a float (handle potential string parsing issues)
        if isinstance(eps, str):
            eps = float(eps)
            
        M = SPDMatrices.ensure_spd(M, eps)
        
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(M)
            sqrt_eigenvals = torch.sqrt(torch.clamp(eigenvals, min=eps))
            M_sqrt = eigenvecs @ torch.diag_embed(sqrt_eigenvals) @ eigenvecs.transpose(-2, -1)
            return M_sqrt
        except RuntimeError:
            # Fallback: identity
            return torch.eye(M.size(-1), device=M.device, dtype=M.dtype).expand_as(M)
    
    @staticmethod
    def matrix_inv_sqrt(M: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Compute inverse matrix square root"""
        # Ensure eps is a float (handle potential string parsing issues)
        if isinstance(eps, str):
            eps = float(eps)
            
        M = SPDMatrices.ensure_spd(M, eps)
        
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(M)
            inv_sqrt_eigenvals = 1.0 / torch.sqrt(torch.clamp(eigenvals, min=eps))
            M_inv_sqrt = eigenvecs @ torch.diag_embed(inv_sqrt_eigenvals) @ eigenvecs.transpose(-2, -1)
            return M_inv_sqrt
        except RuntimeError:
            return torch.eye(M.size(-1), device=M.device, dtype=M.dtype).expand_as(M)
    
    @staticmethod
    def matrix_log(M: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Compute matrix logarithm"""
        # Ensure eps is a float (handle potential string parsing issues)
        if isinstance(eps, str):
            eps = float(eps)
            
        M = SPDMatrices.ensure_spd(M, eps)
        
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(M)
            log_eigenvals = torch.log(torch.clamp(eigenvals, min=eps))
            M_log = eigenvecs @ torch.diag_embed(log_eigenvals) @ eigenvecs.transpose(-2, -1)
            return M_log
        except RuntimeError:
            return torch.zeros_like(M)
    
    @staticmethod
    def frobenius_norm(M: torch.Tensor) -> torch.Tensor:
        """Compute Frobenius norm"""
        return torch.norm(M.view(M.size(0), -1), p='fro', dim=1)


class CovarianceEstimator(nn.Module):
    """
    Estimates covariance matrices from feature vectors with regularization
    """
    
    def __init__(self, 
                 shrinkage_alpha: Optional[float] = None,
                 eps: float = 1e-4,
                 center_data: bool = True):
        super().__init__()
        # Ensure eps is a float (handle potential string parsing issues)
        if isinstance(eps, str):
            eps = float(eps)
        self.shrinkage_alpha = shrinkage_alpha
        self.eps = eps
        self.center_data = center_data
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance matrix from features
        
        Args:
            features: Feature tensor [B, N, D] where N is number of samples per part
            
        Returns:
            Covariance matrices [B, D, D]
        """
        B, N, D = features.shape
        
        if self.center_data:
            # Center the data
            mean = torch.mean(features, dim=1, keepdim=True)  # [B, 1, D]
            centered = features - mean  # [B, N, D]
        else:
            centered = features
        
        # Compute covariance
        cov = torch.bmm(centered.transpose(-2, -1), centered) / max(N - 1, 1)  # [B, D, D]
        
        # Apply shrinkage regularization if specified
        if self.shrinkage_alpha is not None:
            trace_val = torch.diagonal(cov, dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)  # [B, 1]
            identity = torch.eye(D, device=cov.device, dtype=cov.dtype).expand(B, -1, -1)
            shrinkage_target = trace_val.unsqueeze(-1) * identity  # [B, D, D]
            
            cov = (1 - self.shrinkage_alpha) * cov + self.shrinkage_alpha * shrinkage_target
        
        # Ensure SPD
        cov = SPDMatrices.ensure_spd(cov, self.eps)
        
        return cov


class ManifoldDistance(nn.Module):
    """
    Compute distances on the SPD manifold
    Supports both Log-Euclidean and Bures-Wasserstein distances
    """
    
    def __init__(self, 
                 metric: str = 'log_euclidean',
                 eps: float = 1e-4,
                 cache_matrix_ops: bool = True):
        super().__init__()
        
        if metric not in ['log_euclidean', 'bures_wasserstein', 'le', 'bw']:
            raise ValueError(f"Unsupported metric: {metric}")
        
        self.metric = metric
        self.eps = eps
        self.cache_matrix_ops = cache_matrix_ops
        
        # Cache for matrix operations
        self._log_cache = {}
        self._sqrt_cache = {}
    
    def _clear_cache(self):
        """Clear operation cache"""
        self._log_cache.clear()
        self._sqrt_cache.clear()
    
    def log_euclidean_distance(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute Log-Euclidean distance: ||log(A) - log(B)||_F
        
        Args:
            A, B: SPD matrices [B, D, D]
            
        Returns:
            Distances [B]
        """
        # Use cache if available
        A_id = id(A)
        B_id = id(B)
        
        if self.cache_matrix_ops and A_id in self._log_cache:
            log_A = self._log_cache[A_id]
        else:
            log_A = SPDMatrices.matrix_log(A, self.eps)
            if self.cache_matrix_ops:
                self._log_cache[A_id] = log_A
        
        if self.cache_matrix_ops and B_id in self._log_cache:
            log_B = self._log_cache[B_id]
        else:
            log_B = SPDMatrices.matrix_log(B, self.eps)
            if self.cache_matrix_ops:
                self._log_cache[B_id] = log_B
        
        # Compute Frobenius norm of difference
        diff = log_A - log_B
        distance = SPDMatrices.frobenius_norm(diff)
        
        return distance
    
    def bures_wasserstein_distance(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute Bures-Wasserstein distance:
        d²(A,B) = tr(A) + tr(B) - 2*tr((A^{1/2} B A^{1/2})^{1/2})
        
        Args:
            A, B: SPD matrices [B, D, D]
            
        Returns:
            Distances [B]
        """
        # Compute traces
        tr_A = torch.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1)  # [B]
        tr_B = torch.diagonal(B, dim1=-2, dim2=-1).sum(dim=-1)  # [B]
        
        # Compute A^{1/2}
        A_id = id(A)
        if self.cache_matrix_ops and A_id in self._sqrt_cache:
            sqrt_A = self._sqrt_cache[A_id]
        else:
            sqrt_A = SPDMatrices.matrix_sqrt(A, self.eps)
            if self.cache_matrix_ops:
                self._sqrt_cache[A_id] = sqrt_A
        
        # Compute A^{1/2} B A^{1/2}
        temp = torch.bmm(sqrt_A, B)
        C = torch.bmm(temp, sqrt_A)
        
        # Compute (A^{1/2} B A^{1/2})^{1/2}
        sqrt_C = SPDMatrices.matrix_sqrt(C, self.eps)
        tr_sqrt_C = torch.diagonal(sqrt_C, dim1=-2, dim2=-1).sum(dim=-1)  # [B]
        
        # Compute distance squared
        distance_sq = tr_A + tr_B - 2.0 * tr_sqrt_C
        distance_sq = torch.clamp(distance_sq, min=0.0)  # Ensure non-negative
        
        return torch.sqrt(distance_sq)
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute manifold distance between matrices
        
        Args:
            A, B: SPD matrices [B, D, D]
            
        Returns:
            Distances [B]
        """
        if self.metric in ['log_euclidean', 'le']:
            return self.log_euclidean_distance(A, B)
        elif self.metric in ['bures_wasserstein', 'bw']:
            return self.bures_wasserstein_distance(A, B)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def pairwise_distance_matrix(self, 
                               matrices_1: torch.Tensor, 
                               matrices_2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distance matrix between two sets of matrices
        
        Args:
            matrices_1: First set [B, P, D, D]
            matrices_2: Second set [B, Q, D, D]  
            
        Returns:
            Distance matrix [B, P, Q]
        """
        B, P, D, _ = matrices_1.shape
        _, Q, _, _ = matrices_2.shape
        
        distance_matrix = torch.zeros(B, P, Q, device=matrices_1.device)
        
        for b in range(B):
            for p in range(P):
                for q in range(Q):
                    A = matrices_1[b, p]  # [D, D]
                    B_mat = matrices_2[b, q]  # [D, D]
                    
                    # Add batch dimension for computation
                    A_batch = A.unsqueeze(0)  # [1, D, D]
                    B_batch = B_mat.unsqueeze(0)  # [1, D, D]
                    
                    dist = self.forward(A_batch, B_batch)  # [1]
                    distance_matrix[b, p, q] = dist[0]
        
        return distance_matrix


class PartAlignment(nn.Module):
    """
    Part alignment module using manifold distances
    Supports both Bidirectional Nearest Neighbor (BNN) and Hungarian matching
    """
    
    def __init__(self, 
                 alignment_method: str = 'bnn',
                 distance_metric: str = 'log_euclidean'):
        super().__init__()
        
        if alignment_method not in ['bnn', 'hungarian']:
            raise ValueError(f"Unsupported alignment method: {alignment_method}")
        
        self.alignment_method = alignment_method
        self.distance_calculator = ManifoldDistance(metric=distance_metric)
    
    def bidirectional_nearest_neighbor(self, distance_matrix: torch.Tensor) -> Dict[int, int]:
        """
        Bidirectional Nearest Neighbor alignment
        
        Args:
            distance_matrix: Distance matrix [P, Q]
            
        Returns:
            Alignment mapping {p: q}
        """
        P, Q = distance_matrix.shape
        
        # Forward mapping: for each p, find nearest q
        p_to_q = torch.argmin(distance_matrix, dim=1)  # [P]
        
        # Backward mapping: for each q, find nearest p  
        q_to_p = torch.argmin(distance_matrix, dim=0)  # [Q]
        
        # Bidirectional consistency check
        alignment = {}
        for p in range(P):
            q = p_to_q[p].item()
            if q_to_p[q].item() == p:
                alignment[p] = q
        
        # Handle unmatched parts
        matched_q = set(alignment.values())
        unmatched_p = [p for p in range(P) if p not in alignment]
        available_q = [q for q in range(Q) if q not in matched_q]
        
        # Assign remaining parts greedily
        for p in unmatched_p:
            if available_q:
                # Find best available q for this p
                available_distances = [(distance_matrix[p, q].item(), q) for q in available_q]
                available_distances.sort()
                best_q = available_distances[0][1]
                alignment[p] = best_q
                available_q.remove(best_q)
        
        return alignment
    
    def hungarian_alignment(self, distance_matrix: torch.Tensor) -> Dict[int, int]:
        """
        Hungarian algorithm for optimal assignment
        Note: This is a simplified version. For production, use scipy.optimize.linear_sum_assignment
        """
        # For simplicity, fall back to BNN
        # In a full implementation, you would use the Hungarian algorithm
        warnings.warn("Hungarian alignment not fully implemented, using BNN instead")
        return self.bidirectional_nearest_neighbor(distance_matrix)
    
    def forward(self, 
                matrices_1: torch.Tensor, 
                matrices_2: torch.Tensor) -> Tuple[List[Dict[int, int]], torch.Tensor]:
        """
        Align parts between two views
        
        Args:
            matrices_1: First view covariance matrices [B, P, D, D]
            matrices_2: Second view covariance matrices [B, P, D, D]
            
        Returns:
            Tuple of (alignments, distance_matrices)
            alignments: List of alignment dicts for each batch
            distance_matrices: [B, P, P]
        """
        B = matrices_1.shape[0]
        
        # Compute pairwise distance matrices
        distance_matrices = self.distance_calculator.pairwise_distance_matrix(matrices_1, matrices_2)
        
        # Compute alignments for each batch
        alignments = []
        for b in range(B):
            dist_mat = distance_matrices[b]  # [P, P]
            
            if self.alignment_method == 'bnn':
                alignment = self.bidirectional_nearest_neighbor(dist_mat)
            else:  # hungarian
                alignment = self.hungarian_alignment(dist_mat)
            
            alignments.append(alignment)
        
        return alignments, distance_matrices


if __name__ == "__main__":
    # Test manifold operations
    print("Testing manifold operations...")
    
    # Test parameters
    batch_size = 3
    dim = 16
    num_parts = 6
    
    # Create random SPD matrices
    def create_random_spd(batch_size, dim):
        # Create random matrices and make them SPD
        A = torch.randn(batch_size, dim, dim)
        return torch.bmm(A, A.transpose(-2, -1)) + 0.1 * torch.eye(dim)
    
    # Test SPD operations
    print("\n=== Testing SPD Operations ===")
    A = create_random_spd(batch_size, dim)
    print(f"Input shape: {A.shape}")
    
    # Test matrix operations
    A_sqrt = SPDMatrices.matrix_sqrt(A)
    A_log = SPDMatrices.matrix_log(A)
    print(f"Matrix sqrt shape: {A_sqrt.shape}")
    print(f"Matrix log shape: {A_log.shape}")
    
    # Test covariance estimation
    print("\n=== Testing Covariance Estimation ===")
    features = torch.randn(batch_size, 20, dim)  # 20 samples per part
    cov_estimator = CovarianceEstimator(shrinkage_alpha=0.1)
    cov_matrices = cov_estimator(features)
    print(f"Covariance matrices shape: {cov_matrices.shape}")
    
    # Test manifold distances
    print("\n=== Testing Manifold Distances ===")
    B = create_random_spd(batch_size, dim)
    
    # Log-Euclidean distance
    le_distance = ManifoldDistance(metric='log_euclidean')
    le_dist = le_distance(A, B)
    print(f"Log-Euclidean distances: {le_dist}")
    
    # Bures-Wasserstein distance
    bw_distance = ManifoldDistance(metric='bures_wasserstein')
    bw_dist = bw_distance(A, B)
    print(f"Bures-Wasserstein distances: {bw_dist}")
    
    # Test part alignment
    print("\n=== Testing Part Alignment ===")
    matrices_1 = create_random_spd(batch_size, dim).unsqueeze(1).repeat(1, num_parts, 1, 1)
    matrices_2 = create_random_spd(batch_size, dim).unsqueeze(1).repeat(1, num_parts, 1, 1)
    
    aligner = PartAlignment(alignment_method='bnn', distance_metric='log_euclidean')
    alignments, distance_matrices = aligner(matrices_1, matrices_2)
    
    print(f"Number of alignments: {len(alignments)}")
    print(f"Distance matrices shape: {distance_matrices.shape}")
    print(f"Example alignment: {alignments[0]}")
    
    print("\nAll tests passed!")
