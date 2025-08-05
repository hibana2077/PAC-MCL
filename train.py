"""
Main training script for PAC-MCL
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
import os
import sys
from pathlib import Path
import time
import random
import numpy as np
from typing import Dict, Any, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models import PAC_MCL_Model
from src.dataset.ufgvc import UFGVCDataset
from src.utils import (
    create_train_transforms, create_val_transforms,
    MetricsCalculator, ModelEvaluator,
    EarlyStopping, LearningRateScheduler, ModelCheckpoint, TrainingTracker
)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    # Create transforms
    if config['training']['dual_view']:
        train_transform = create_train_transforms(
            image_size=config['data']['image_size'],
            dual_view=True,
            use_strong_aug=True
        )
    else:
        train_transform = create_train_transforms(
            image_size=config['data']['image_size'],
            dual_view=False,
            use_strong_aug=False
        )
    
    val_transform = create_val_transforms(
        image_size=config['data']['image_size']
    )
    
    # Create datasets
    train_dataset = UFGVCDataset(
        dataset_name=config['data']['dataset_name'],
        root=config['data']['data_root'],
        split='train',
        transform=train_transform,
        download=True
    )
    
    val_dataset = UFGVCDataset(
        dataset_name=config['data']['dataset_name'],
        root=config['data']['data_root'],
        split='val',
        transform=val_transform,
        download=False
    )
    
    test_dataset = UFGVCDataset(
        dataset_name=config['data']['dataset_name'],
        root=config['data']['data_root'],
        split='test',
        transform=val_transform,
        download=False
    )
    
    # Update num_classes in config
    config['model']['num_classes'] = len(train_dataset.classes)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_model_and_optimizer(config: Dict[str, Any], device: torch.device):
    """Create model, optimizer, and scheduler"""
    
    # Create model
    model = PAC_MCL_Model(**config['model'])
    model.to(device)
    
    # Create optimizer
    if config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.999)
        )
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
    
    # Create scheduler
    scheduler = None
    if config['training']['scheduler']['type'] != 'none':
        scheduler = LearningRateScheduler.create_scheduler(
            config['training']['scheduler']['type'],
            optimizer,
            **config['training']['scheduler']['params']
        )
    
    return model, optimizer, scheduler


def train_epoch(model: PAC_MCL_Model,
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                dual_view: bool = True) -> Dict[str, float]:
    """Train for one epoch"""
    
    model.train()
    metrics_calc = MetricsCalculator(model.num_classes)
    
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, batch_data in enumerate(train_loader):
        
        if dual_view:
            # Dual view training
            (images_v1, images_v2), labels = batch_data
            images_v1 = images_v1.to(device)
            images_v2 = images_v2.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images_v1, images_v2, labels, return_loss=True)
            loss = outputs['loss']
            logits = outputs['logits_v1']  # Use first view for metrics
            
        else:
            # Single view training (fallback)
            images, labels = batch_data
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model.forward_inference(images)
            logits = outputs['logits']
            
            # Compute classification loss
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)
        metrics_calc.update(predictions, labels, probabilities, loss.item())
        
        total_loss += loss.item()
        
        # Log progress
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
    
    # Compute epoch metrics
    metrics = metrics_calc.compute_basic_metrics()
    metrics['loss'] = total_loss / num_batches
    
    return metrics


def validate_epoch(model: PAC_MCL_Model,
                  val_loader: DataLoader,
                  device: torch.device) -> Dict[str, float]:
    """Validate for one epoch"""
    
    evaluator = ModelEvaluator(model, device, model.num_classes)
    metrics = evaluator.evaluate_dataloader(val_loader)
    
    return {
        'loss': metrics['avg_loss'],
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'top5_accuracy': metrics.get('top5_accuracy', 0.0)
    }


def main():
    parser = argparse.ArgumentParser(description='Train PAC-MCL model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    print(f"Dataset: {config['data']['dataset_name']}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Num classes: {config['model']['num_classes']}")
    
    # Create model, optimizer, scheduler
    print("Creating model...")
    model, optimizer, scheduler = create_model_and_optimizer(config, device)
    
    model_info = model.get_model_info()
    print(f"Model: {model_info['backbone_name']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # Create training utilities
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta'],
        mode='max',  # Monitor accuracy
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        checkpoint_dir=str(checkpoint_dir),
        save_best=True,
        save_last=True,
        monitor='val_accuracy',
        mode='max',
        save_top_k=3
    )
    
    tracker = TrainingTracker(log_dir=str(log_dir))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint_data = checkpoint.load_checkpoint(
            args.resume, model, optimizer, scheduler, map_location=device
        )
        start_epoch = checkpoint_data['epoch'] + 1
    
    # Training loop
    print(f"Starting training for {config['training']['epochs']} epochs...")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start_time = time.time()
        
        # Set current epoch for potential curriculum learning
        model.set_epoch(epoch)
        
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            dual_view=config['training']['dual_view']
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device)
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Track training
        tracker.log_epoch(epoch, epoch_time, train_metrics, val_metrics)
        
        # Save checkpoint
        all_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
        all_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
        
        checkpoint.save_checkpoint(
            model, optimizer, scheduler, epoch, all_metrics,
            additional_info={'config': config, 'args': vars(args)}
        )
        
        # Early stopping check
        if early_stopping(val_metrics['accuracy'], model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final evaluation on test set...")
    
    # Load best model
    best_checkpoint_path = checkpoint.get_best_checkpoint()
    if best_checkpoint_path:
        checkpoint.load_checkpoint(best_checkpoint_path, model, map_location=device)
        print("Loaded best model for final evaluation")
    
    # Test evaluation
    test_metrics = validate_epoch(model, test_loader, device)
    
    print("Test Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Top-5 Accuracy: {test_metrics['top5_accuracy']:.4f}")
    
    # Create detailed evaluation report
    evaluator = ModelEvaluator(model, device, model.num_classes, train_loader.dataset.classes)
    
    # Classification report
    report = evaluator.create_classification_report(
        test_loader, 
        save_path=output_dir / 'classification_report.txt'
    )
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(
        test_loader,
        save_path=output_dir / 'confusion_matrix.png',
        normalize=True
    )
    
    # Plot training metrics
    tracker.plot_metrics(save_path=output_dir / 'training_metrics.png')
    
    # Save final results
    final_results = {
        'config': config,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'model_info': model_info,
        'training_completed': True
    }
    
    import json
    with open(output_dir / 'final_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        final_results = convert_numpy(final_results)
        json.dump(final_results, f, indent=2)
    
    print(f"\nTraining completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
