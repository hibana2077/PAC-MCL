# PAC-MCL Usage Guide

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Training

Train the PAC-MCL model with default configuration:

```bash
python train.py --config configs/cotton80_resnet50.yaml
```

Available configurations:
- `configs/cotton80_resnet50.yaml` - PAC-MCL on Cotton-80 with ResNet-50
- `configs/soybean_convnext.yaml` - PAC-MCL on Soybean with ConvNeXt
- `configs/baseline_cotton80.yaml` - Baseline model without PAC-MCL

### 3. Evaluation

Evaluate a trained model:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --config configs/cotton80_resnet50.yaml
```

## Configuration

### Model Settings
- `model.backbone`: CNN backbone (resnet50, convnext_tiny, efficientnet_b3, etc.)
- `model.num_parts`: Number of parts to extract (default: 4)
- `model.part_size`: Size of each part region (default: 64)
- `model.feature_dim`: Feature dimension (default: 512)

### Training Settings
- `training.epochs`: Number of training epochs
- `training.batch_size`: Batch size for training
- `training.learning_rate`: Initial learning rate
- `training.weight_decay`: Weight decay for regularization

### Loss Settings
- `loss.alpha`: Weight for part-level contrastive loss
- `loss.beta`: Weight for manifold alignment loss
- `loss.temperature`: Temperature for contrastive learning
- `loss.margin`: Margin for contrastive loss

## Dataset Structure

Ensure your dataset follows this structure:
```
data/
├── Cotton-80/
│   ├── images/
│   ├── train_split.txt
│   ├── val_split.txt
│   └── test_split.txt
└── Soybean/
    ├── images/
    ├── train_split.txt
    ├── val_split.txt
    └── test_split.txt
```

## Model Architecture

The PAC-MCL model consists of:

1. **Backbone**: CNN feature extractor (using timm)
2. **Part Extractor**: Attention-based part region extraction
3. **Manifold Operations**: SPD matrix operations for covariance features
4. **Contrastive Learning**: Part-aware contrastive loss with manifold alignment

## Training Process

1. **Dual-view Generation**: Strong augmentations create two views per image
2. **Part Extraction**: Attention mechanism identifies discriminative parts
3. **Feature Extraction**: CNN backbone extracts features for each part
4. **Manifold Learning**: Covariance matrices represent part relationships
5. **Alignment**: Hungarian algorithm aligns parts between views
6. **Loss Computation**: Combined classification and contrastive losses

## Experiment Tracking

The training script logs:
- Training/validation accuracy and loss
- Part-level contrastive loss
- Manifold alignment loss
- Learning rate schedules
- Model checkpoints

## Output Files

Training generates:
- `checkpoints/`: Saved model checkpoints
- `logs/`: Training logs and metrics
- `results/`: Evaluation results and visualizations

## Tips

1. **GPU Memory**: Reduce batch size if encountering OOM errors
2. **Learning Rate**: Use smaller LR for fine-tuning pre-trained models
3. **Augmentations**: Adjust augmentation strength based on dataset complexity
4. **Parts**: Increase num_parts for more complex fine-grained categories
