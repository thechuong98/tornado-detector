"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.


The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

"""
CoordConv for tornado detection
"""

# import torch
# from torch import nn
# import torch.nn.modules.conv as conv
from typing import Dict, List, Tuple, Any
import numpy as np
# import torch
# from torch import nn, optim
# from torch.optim.lr_scheduler import StepLR
# import torch.nn.functional as F
# from torchmetrics import MetricCollection
# import lightning as L
from scipy.ndimage import label, center_of_mass
import onnx
import onnxruntime as ort

from src.constant import CHANNEL_MIN_MAX, ALL_VARIABLES


def bilinear_interpolation(input_array, output_shape):
        """
        Performs bilinear interpolation to resize an array to the desired shape.
        
        Args:
            input_array: numpy array of shape (h1, w1)
            output_shape: tuple of (h2, w2) for desired output shape
        
        Returns:
            Interpolated array of shape output_shape
        """
        input_height, input_width = input_array.shape
        output_height, output_width = output_shape
        
        # Create coordinate matrices for the output grid
        x = np.linspace(0, input_width - 1, output_width)
        y = np.linspace(0, input_height - 1, output_height)
        
        # Get the integer and fractional parts of the coordinates
        x0 = np.floor(x).astype(int)
        x1 = np.minimum(x0 + 1, input_width - 1)
        y0 = np.floor(y).astype(int)
        y1 = np.minimum(y0 + 1, input_height - 1)
        
        # Get the fractional parts
        xf = x - x0
        yf = y - y0
        
        # Reshape for broadcasting
        xf = xf.reshape(1, -1)
        yf = yf.reshape(-1, 1)
        
        # Get the four corner values for each output point
        p00 = input_array[y0][:, x0]
        p01 = input_array[y0][:, x1]
        p10 = input_array[y1][:, x0]
        p11 = input_array[y1][:, x1]
        
        # Perform bilinear interpolation
        output = (p00 * (1 - xf) * (1 - yf) +
                 p01 * xf * (1 - yf) +
                 p10 * (1 - xf) * yf +
                 p11 * xf * yf)
        
        return output

class TornadoDetectionModel:
    def __init__(self, model_path: str, num_range: int, include_range_folded: bool = True):
        self.num_range = num_range
        # self.model = TornadoLikelihood(
        #     shape=(2, 720, self.num_range),
        #     c_shape=(2, 720, self.num_range),
        #     input_variables=['DBZ',
        #         'VEL', 
        #         'KDP',
        #         'RHOHV',
        #         'ZDR',
        #         'WIDTH'],
        #     start_filters=48,
        #     background_flag=-3.0,
        #     include_range_folded=include_range_folded
        # )
        try:
            self.model = ort.InferenceSession(model_path)
        # try:
        #     state_dict = torch.load(model_path)
        #     self.model.load_state_dict(state_dict)
        #     self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
    def detect_tornados(self, probability_map, threshold=0.5):
        """Detect potential tornado locations from probability map"""
        # Threshold the probability map
        binary_map = probability_map > threshold
        
        # Label connected components
        labeled_array, num_features = label(binary_map)
        # Get centroids and sizes of each connected component
        centroids = []
        probabilities = []
        sizes = []
        for i in range(1, num_features + 1):
            mask = labeled_array == i
            if np.sum(mask) > 0:  # Ensure the region exists
                y, x = center_of_mass(mask)
                centroids.append((x, y))
                probabilities.append(np.max(probability_map[mask]))
                sizes.append(np.sum(mask))  # Count number of pixels in cluster
        
        return centroids, probabilities, sizes
    
    def calculate_likelihood(self, example):
        infer_var = ['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH', 'range_folded_mask', 'coordinates']

        infer_data = {}
        for var in infer_var:
            # Transpose from (batch, height, width, channel) to (batch, channel, height, width)
            infer_data[var] = np.transpose(example[var], (0, 3, 1, 2)).astype(np.float32)

        # Run inference with ONNX model
        output = self.model.run(None, infer_data)[0]
        return output

    def predict(self, example):
        likelihood = self.calculate_likelihood(example)
        # Global max pooling
        probability = np.max(likelihood)
        # Apply sigmoid
        probability = 1/(1 + np.exp(-probability))
        return float(probability)

