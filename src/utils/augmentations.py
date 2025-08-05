"""
Strong augmentation strategies for PAC-MCL
Implements CLE-style masking and shuffling augmentations for dual-view generation
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter
import random
import numpy as np
from typing import Tuple, List, Optional, Union, Callable
import math


class RandomMasking:
    """Random masking augmentation"""
    
    def __init__(self, 
                 mask_ratio: float = 0.3,
                 min_mask_size: int = 16,
                 max_mask_size: int = 64):
        self.mask_ratio = mask_ratio
        self.min_mask_size = min_mask_size
        self.max_mask_size = max_mask_size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random masking to image"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Calculate number of masks based on ratio
        total_pixels = h * w
        masked_pixels = int(total_pixels * self.mask_ratio)
        
        # Generate random masks
        current_masked = 0
        while current_masked < masked_pixels:
            # Random mask size
            mask_size = random.randint(self.min_mask_size, self.max_mask_size)
            
            # Random position
            x = random.randint(0, max(0, w - mask_size))
            y = random.randint(0, max(0, h - mask_size))
            
            # Apply mask (set to random color or mean)
            mask_color = np.random.randint(0, 256, 3) if len(img_array.shape) == 3 else np.random.randint(0, 256)
            img_array[y:y+mask_size, x:x+mask_size] = mask_color
            
            current_masked += mask_size * mask_size
        
        return Image.fromarray(img_array)


class BlockShuffling:
    """Block shuffling augmentation"""
    
    def __init__(self, 
                 block_size: int = 32,
                 shuffle_ratio: float = 0.5):
        self.block_size = block_size
        self.shuffle_ratio = shuffle_ratio
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply block shuffling to image"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Calculate number of blocks
        blocks_h = h // self.block_size
        blocks_w = w // self.block_size
        
        if blocks_h < 2 or blocks_w < 2:
            return img  # Skip if too small
        
        # Extract blocks
        blocks = []
        positions = []
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                y1, y2 = i * self.block_size, (i + 1) * self.block_size
                x1, x2 = j * self.block_size, (j + 1) * self.block_size
                
                block = img_array[y1:y2, x1:x2].copy()
                blocks.append(block)
                positions.append((i, j))
        
        # Shuffle a portion of blocks
        num_shuffle = int(len(blocks) * self.shuffle_ratio)
        shuffle_indices = random.sample(range(len(blocks)), num_shuffle)
        
        if len(shuffle_indices) >= 2:
            # Randomly shuffle selected blocks
            shuffled_blocks = [blocks[i] for i in shuffle_indices]
            random.shuffle(shuffled_blocks)
            
            # Place shuffled blocks back
            for idx, shuffled_block in zip(shuffle_indices, shuffled_blocks):
                i, j = positions[idx]
                y1, y2 = i * self.block_size, (i + 1) * self.block_size
                x1, x2 = j * self.block_size, (j + 1) * self.block_size
                img_array[y1:y2, x1:x2] = shuffled_block
        
        return Image.fromarray(img_array)


class CutMix:
    """CutMix-style augmentation"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply CutMix to image (self-mixing)"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Generate random box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Random position
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Create flipped/rotated version of the same image
        transform_img = img_array.copy()
        transform_img = np.fliplr(transform_img)  # Horizontal flip
        
        # Mix the regions
        img_array[bby1:bby2, bbx1:bbx2] = transform_img[bby1:bby2, bbx1:bbx2]
        
        return Image.fromarray(img_array)


class GridMask:
    """GridMask augmentation"""
    
    def __init__(self, 
                 d_range: Tuple[int, int] = (96, 224),
                 rotate: int = 1,
                 ratio: float = 0.6):
        self.d_range = d_range
        self.rotate = rotate
        self.ratio = ratio
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply GridMask to image"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Random grid size
        d = random.randint(self.d_range[0], self.d_range[1])
        
        # Create grid mask
        mask = np.ones((h, w), dtype=np.float32)
        
        # Grid parameters
        l = int(d * self.ratio)  # Length of mask
        
        for i in range(0, h, d):
            for j in range(0, w, d):
                # Apply mask in grid pattern
                y1, y2 = i, min(i + l, h)
                x1, x2 = j, min(j + l, w)
                mask[y1:y2, x1:x2] = 0
        
        # Apply rotation if specified
        if self.rotate:
            angle = random.randint(-45, 45)
            from scipy.ndimage import rotate
            mask = rotate(mask, angle, reshape=False, order=0)
            mask = np.clip(mask, 0, 1)
        
        # Apply mask to image
        if len(img_array.shape) == 3:
            mask = np.expand_dims(mask, axis=2)
        img_array = img_array * mask
        
        return Image.fromarray(img_array.astype(np.uint8))


class StrongAugmentation:
    """
    Strong augmentation pipeline combining multiple techniques
    Inspired by CLE-ViT augmentation strategies
    """
    
    def __init__(self,
                 image_size: int = 224,
                 use_masking: bool = True,
                 use_shuffling: bool = True,
                 use_cutmix: bool = True,
                 use_gridmask: bool = False,
                 mask_ratio: float = 0.3,
                 shuffle_ratio: float = 0.5,
                 augment_prob: float = 0.8):
        
        self.image_size = image_size
        self.augment_prob = augment_prob
        
        # Initialize augmentation techniques
        self.augmentations = []
        
        if use_masking:
            self.augmentations.append(RandomMasking(mask_ratio=mask_ratio))
        
        if use_shuffling:
            self.augmentations.append(BlockShuffling(shuffle_ratio=shuffle_ratio))
        
        if use_cutmix:
            self.augmentations.append(CutMix(alpha=1.0))
        
        if use_gridmask:
            self.augmentations.append(GridMask())
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply strong augmentation"""
        if random.random() < self.augment_prob and self.augmentations:
            # Randomly select one augmentation technique
            aug = random.choice(self.augmentations)
            img = aug(img)
        
        return img


def create_dual_view_transform(base_transform: transforms.Compose,
                              strong_aug: StrongAugmentation) -> Callable:
    """
    Create a transform that generates two views of the same image
    
    Args:
        base_transform: Base transformation pipeline
        strong_aug: Strong augmentation for positive view
        
    Returns:
        Transform function that returns (view1, view2)
    """
    def dual_transform(img):
        # Apply base transform to get two different random augmentations
        view1 = base_transform(img)
        
        # Apply strong augmentation + base transform for second view
        strong_img = strong_aug(img)
        view2 = base_transform(strong_img)
        
        return view1, view2
    
    return dual_transform


def create_train_transforms(image_size: int = 224,
                          mean: List[float] = [0.485, 0.456, 0.406],
                          std: List[float] = [0.229, 0.224, 0.225],
                          use_strong_aug: bool = True,
                          dual_view: bool = False) -> Union[transforms.Compose, Callable]:
    """
    Create training transforms
    
    Args:
        image_size: Target image size
        mean, std: Normalization parameters
        use_strong_aug: Whether to use strong augmentation
        dual_view: Whether to return dual views
        
    Returns:
        Transform composition or dual-view transform function
    """
    # Base augmentations
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    
    base_transform = transforms.Compose(base_transforms)
    
    if dual_view:
        # Create strong augmentation
        strong_aug = StrongAugmentation(
            image_size=image_size,
            use_masking=use_strong_aug,
            use_shuffling=use_strong_aug,
            use_cutmix=use_strong_aug
        )
        
        return create_dual_view_transform(base_transform, strong_aug)
    
    return base_transform


def create_val_transforms(image_size: int = 224,
                         mean: List[float] = [0.485, 0.456, 0.406],
                         std: List[float] = [0.229, 0.224, 0.225]) -> transforms.Compose:
    """Create validation/test transforms"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


class TensorAugmentation:
    """
    Tensor-based augmentations that can be applied on GPU
    """
    
    @staticmethod
    def random_masking_tensor(x: torch.Tensor, 
                            mask_ratio: float = 0.3,
                            mask_value: float = 0.0) -> torch.Tensor:
        """Apply random masking to tensor"""
        B, C, H, W = x.shape
        
        # Create random mask
        mask = torch.rand(B, 1, H, W, device=x.device) > mask_ratio
        
        # Apply mask
        masked_x = x * mask + mask_value * (1 - mask)
        
        return masked_x
    
    @staticmethod
    def random_cutout(x: torch.Tensor,
                     cutout_size: int = 16,
                     num_cutouts: int = 1) -> torch.Tensor:
        """Apply random cutout to tensor"""
        B, C, H, W = x.shape
        
        for _ in range(num_cutouts):
            # Random position
            y = torch.randint(0, H - cutout_size + 1, (B,))
            x_pos = torch.randint(0, W - cutout_size + 1, (B,))
            
            for b in range(B):
                x[b, :, y[b]:y[b]+cutout_size, x_pos[b]:x_pos[b]+cutout_size] = 0
        
        return x


if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentation pipeline...")
    
    # Create test image
    test_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    # Test individual augmentations
    print("\n=== Testing Individual Augmentations ===")
    
    # Random masking
    masking = RandomMasking(mask_ratio=0.3)
    masked_img = masking(test_img)
    print("Random masking: OK")
    
    # Block shuffling
    shuffling = BlockShuffling(block_size=32, shuffle_ratio=0.5)
    shuffled_img = shuffling(test_img)
    print("Block shuffling: OK")
    
    # CutMix
    cutmix = CutMix(alpha=1.0)
    cutmix_img = cutmix(test_img)
    print("CutMix: OK")
    
    # Strong augmentation
    print("\n=== Testing Strong Augmentation ===")
    strong_aug = StrongAugmentation(
        image_size=224,
        use_masking=True,
        use_shuffling=True,
        use_cutmix=True
    )
    
    strong_img = strong_aug(test_img)
    print("Strong augmentation: OK")
    
    # Test transform creation
    print("\n=== Testing Transform Creation ===")
    
    # Single view transform
    train_transform = create_train_transforms(
        image_size=224,
        use_strong_aug=True,
        dual_view=False
    )
    
    single_tensor = train_transform(test_img)
    print(f"Single view tensor shape: {single_tensor.shape}")
    
    # Dual view transform
    dual_transform = create_train_transforms(
        image_size=224,
        use_strong_aug=True,
        dual_view=True
    )
    
    view1, view2 = dual_transform(test_img)
    print(f"Dual view tensors shape: {view1.shape}, {view2.shape}")
    
    # Validation transform
    val_transform = create_val_transforms(image_size=224)
    val_tensor = val_transform(test_img)
    print(f"Validation tensor shape: {val_tensor.shape}")
    
    # Test tensor augmentations
    print("\n=== Testing Tensor Augmentations ===")
    test_tensor = torch.randn(2, 3, 224, 224)
    
    masked_tensor = TensorAugmentation.random_masking_tensor(test_tensor, mask_ratio=0.3)
    print(f"Masked tensor shape: {masked_tensor.shape}")
    
    cutout_tensor = TensorAugmentation.random_cutout(test_tensor, cutout_size=32)
    print(f"Cutout tensor shape: {cutout_tensor.shape}")
    
    print("\nAll augmentation tests passed!")
