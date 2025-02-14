
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pytest
import numpy as np
from src.model import TornadoLikelihood
from src.unet_vda_model import VelocityDealiaser
import torch


model = TornadoLikelihood(
    shape=(2, 720, 1152),
    c_shape=(2, 720, 1152),
    input_variables=['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH'],
    start_filters=48,
    background_flag=-3.0,
    include_range_folded=True
)
state_dict = torch.load("model_checkpoint/tornet.pth")
model.load_state_dict(state_dict)
model.eval()


velocity_dealiaser = VelocityDealiaser(
    model_path="model_checkpoint/unet_vda.onnx"
)

def test_velocity_dealiaser():
    # Test velocity dealiasing with valid input
    velocity = np.random.random((1, 720, 1152, 1)) # [batch,tilt,az,rng,1]
    nyquist = np.random.random((1)) # [batch,frames,1]
    
    dealiased = velocity_dealiaser.dealias(velocity, nyquist)
    
    print(dealiased.shape)
    # assert dealiased.shape == (2, 360, 1152)
    # assert not np.any(np.isnan(dealiased))

# def test_velocity_dealiaser_720():
#     # Test with 720 azimuths
#     velocity = np.random.random((1, 2, 720, 1152, 1))
#     nyquist = np.random.random((1, 1, 1))
    
#     dealiased = velocity_dealiaser.dealias(velocity, nyquist)
    
#     assert dealiased.shape == (2, 720, 1152)
#     assert not np.any(np.isnan(dealiased))

# def test_velocity_dealiaser_missing_data():
#     # Test handling of missing data (-64)
#     velocity = np.random.random((1, 2, 360, 1152, 1))
#     velocity[0,0,0,0,0] = -65 # Set one value to missing
#     nyquist = np.random.random((1, 1, 1))
    
#     dealiased = velocity_dealiaser.dealias(velocity, nyquist)
    
#     assert np.isnan(dealiased[0,0,0])

def test_tornado_model():
    # Test tornado likelihood calculation
    example = {
        'DBZ': np.random.random((2, 720, 1152)),
        'VEL': np.random.random((2, 720, 1152)), 
        'KDP': np.random.random((2, 720, 1152)),
        'RHOHV': np.random.random((2, 720, 1152)),
        'ZDR': np.random.random((2, 720, 1152)),
        'WIDTH': np.random.random((2, 720, 1152)),
        'range_folded_mask': np.zeros((2, 720, 1152), dtype=bool),
        'coordinates': np.random.random((2, 720, 1152, 2)),
        'nyquist': np.random.random((1)),
        'az_lower': 0,
        'az_upper': 360,
        'rng_lower': 0,
        'rng_upper': 1152
    }

    result = model.calculate_likelihood(example)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 720, 1152)

def test_tornado_model_missing_field():
    # Test error handling for missing field
    example = {
        'DBZ': np.random.random((2, 720, 1152)),
        # VEL missing
        'KDP': np.random.random((2, 720, 1152)),
        'RHOHV': np.random.random((2, 720, 1152)),
        'ZDR': np.random.random((2, 720, 1152)),
        'WIDTH': np.random.random((2, 720, 1152)),
        'range_folded_mask': np.zeros((2, 720, 1152), dtype=bool),
        'coordinates': np.random.random((2, 720, 1152, 2)),
        'nyquist': np.random.random((1)),
        'az_lower': 0,
        'az_upper': 360,
        'rng_lower': 0,
        'rng_upper': 1152
    }

    with pytest.raises(KeyError):
        model.calculate_likelihood(example)

def test_tornado_model_wrong_shape():
    # Test error handling for incorrect input shape
    example = {
        'DBZ': np.random.random((2, 360, 1152)),  # Wrong azimuth dimension
        'VEL': np.random.random((2, 720, 1152)),
        'KDP': np.random.random((2, 720, 1152)),
        'RHOHV': np.random.random((2, 720, 1152)), 
        'ZDR': np.random.random((2, 720, 1152)),
        'WIDTH': np.random.random((2, 720, 1152)),
        'range_folded_mask': np.zeros((2, 720, 1152), dtype=bool),
        'coordinates': np.random.random((2, 720, 1152, 2)),
        'nyquist': np.random.random((1)),
        'az_lower': 0,
        'az_upper': 360,
        'rng_lower': 0,
        'rng_upper': 1152
    }

    with pytest.raises(ValueError):
        model.calculate_likelihood(example)

if __name__ == "__main__":
    test_velocity_dealiaser()
    # test_velocity_dealiaser_720()
    test_tornado_model()
    test_tornado_model_missing_field()
    test_tornado_model_wrong_shape()