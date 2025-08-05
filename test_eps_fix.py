#!/usr/bin/env python3
"""
Test script to verify eps parameter handling
"""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_yaml_parsing():
    """Test YAML parsing of eps values"""
    configs = [
        'configs/cotton80_resnet50.yaml',
        'configs/soybean_convnext.yaml', 
        'configs/baseline_cotton80.yaml'
    ]
    
    for config_path in configs:
        print(f"\nTesting {config_path}:")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        eps_value = config['model']['eps']
        print(f"  eps value: {eps_value}")
        print(f"  eps type: {type(eps_value)}")
        
        # Test conversion
        if isinstance(eps_value, str):
            converted = float(eps_value)
            print(f"  converted: {converted}")
            print(f"  converted type: {type(converted)}")
        else:
            print(f"  already numeric: {eps_value}")

def test_model_creation():
    """Test model creation with different eps values"""
    from src.models import PAC_MCL_Model
    
    print("\nTesting model creation:")
    
    # Test with float
    try:
        model1 = PAC_MCL_Model(eps=1e-4)
        print("✓ Model creation with float eps: SUCCESS")
    except Exception as e:
        print(f"✗ Model creation with float eps: FAILED - {e}")
    
    # Test with string
    try:
        model2 = PAC_MCL_Model(eps="1e-4")
        print("✓ Model creation with string eps: SUCCESS")
    except Exception as e:
        print(f"✗ Model creation with string eps: FAILED - {e}")
    
    # Test with problematic string
    try:
        model3 = PAC_MCL_Model(eps="0.1e-4")
        print("✓ Model creation with problematic string eps: SUCCESS")
    except Exception as e:
        print(f"✗ Model creation with problematic string eps: FAILED - {e}")

if __name__ == "__main__":
    test_yaml_parsing()
    test_model_creation()
