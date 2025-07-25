import os
import sys
import torch
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.root_morphology_model import RootMorphologyModel
from src.data.preprocess import RootImageDataset
from src.utils.utils import set_seed

def test_model_initialization():
    """Test model initialization"""
    model = RootMorphologyModel(num_classes=10)
    assert isinstance(model, torch.nn.Module)
    assert model.conv1.in_channels == 3
    assert model.conv1.out_channels == 64
    assert model.fc3.out_features == 10

def test_model_forward_pass():
    """Test model forward pass"""
    set_seed(42)
    model = RootMorphologyModel(num_classes=10)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (batch_size, 10)

def test_model_predict():
    """Test model prediction method"""
    set_seed(42)
    model = RootMorphologyModel(num_classes=10)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    predictions = model.predict(input_tensor)
    assert predictions.shape == (batch_size,)
    assert torch.all((predictions >= 0) & (predictions < 10))

def test_model_features():
    """Test feature extraction"""
    set_seed(42)
    model = RootMorphologyModel(num_classes=10)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    features = model.get_features(input_tensor)
    assert features.shape == (batch_size, 512, 14, 14)

def test_dataset_initialization():
    """Test dataset initialization"""
    # Create a temporary directory with some dummy images
    os.makedirs('tests/temp_data', exist_ok=True)
    for i in range(5):
        torch.save(torch.randn(3, 224, 224), f'tests/temp_data/image_{i}.pt')
    
    dataset = RootImageDataset('tests/temp_data')
    assert len(dataset) == 5
    
    # Clean up
    import shutil
    shutil.rmtree('tests/temp_data')

def test_model_save_load():
    """Test model saving and loading"""
    set_seed(42)
    model = RootMorphologyModel(num_classes=10)
    
    # Save model
    os.makedirs('tests/temp_models', exist_ok=True)
    torch.save(model.state_dict(), 'tests/temp_models/test_model.pth')
    
    # Load model
    new_model = RootMorphologyModel(num_classes=10)
    new_model.load_state_dict(torch.load('tests/temp_models/test_model.pth'))
    
    # Compare model parameters
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
    
    # Clean up
    import shutil
    shutil.rmtree('tests/temp_models')

if __name__ == '__main__':
    pytest.main([__file__]) 