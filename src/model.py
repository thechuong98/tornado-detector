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

import torch
from torch import nn
# import torch.nn.modules.conv as conv
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from torch import nn, optim
# from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
# from torchmetrics import MetricCollection
# import lightning as L
from scipy.ndimage import label, center_of_mass

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
        self.model = TornadoLikelihood(
            shape=(2, 720, self.num_range),
            c_shape=(2, 720, self.num_range),
            input_variables=['DBZ',
                'VEL', 
                'KDP',
                'RHOHV',
                'ZDR',
                'WIDTH'],
            start_filters=48,
            background_flag=-3.0,
            include_range_folded=include_range_folded
        )
        try:
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict)
            self.model.eval()
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
            infer_data[var] = example[var]
        infer_data = {k: torch.Tensor(v).permute(0, 3, 1, 2) for k,v in infer_data.items()}
        with torch.no_grad():
            output = self.model(infer_data)
        return output.cpu()
    
    def predict(self, example):
        likelihood = self.calculate_likelihood(example)
        probability = F.adaptive_max_pool2d(likelihood, (1, 1))
        probability = 1/(1+torch.exp(-probability))
        return probability.item()


class CoordConv2D(nn.Module):
    """
    CoordConv2D layer for working with polar data.

    This module takes a tuple of inputs (image tensor,   image coordinates)
    where,
        image tensor is [batch, in_image_channels, height, width]
        image coordinates is [batch, in_coord_channels, height, width]
    
    This returns a tuple containing the CoordConv convolution and 
    a (possibly downsampled) copy of the coordinate tensor.


    """
    def __init__(self,in_image_channels,
                      in_coord_channels,
                      out_channels,
                      kernel_size,
                      padding='same',
                      stride=1,
                      activation='relu',
                      **kwargs):
        super(CoordConv2D, self).__init__()
        self.n_coord_channels=in_coord_channels
        self.conv = nn.Conv2d(in_image_channels + in_coord_channels, 
                              out_channels,
                              kernel_size, 
                              stride, 
                              padding, **kwargs)
        self.strd=stride
        self.padding=padding
        self.ksize=kernel_size
        if activation is None:
            self.conv_activation=None
        elif activation=='relu':
            self.conv_activation=nn.ReLU()
        else:
            raise NotImplementedError('activation %s not implemented' % activation)
        
    
    def forward(self,inputs):
        """
        inputs is a tuple containing 
          (image tensor,   image coordinates)

        image tensor is [batch, in_image_channels, height, width]
        image coordinates is [batch, in_coord_channels, height, width]

        """
        x,coords=inputs
        x = torch.cat( (x,coords),axis=1)
        x = self.conv(x)
        
        # only apply activation to conv output
        if self.conv_activation:
            x=self.conv_activation(x) 

        # also return coordinates
        if self.padding=='same' and self.strd>1:
            coords = coords[...,::self.strd,::self.strd]
        elif self.padding=='valid':
            # If valid padding,  need to start slightly off the corner
            i0 = self.ksize[0]//2
            if i0>0:
                coords = coords[...,i0:-i0:self.strd,i0:-i0:self.strd]
            else:
                coords = coords[...,::self.strd,::self.strd]
        
        return x,coords
        


# class TornadoClassifier(L.LightningModule):
#     """
#     Fits tornado classifier
#     """
#     def __init__(self,model:nn.Module,
#                       lr:float=1e-3,
#                       lr_decay_rate=0.9,
#                       lr_decay_steps=1,
#                       label_smoothing:float=0.2,
#                       weight_decay:float=0.001,
#                       metrics:MetricCollection=None):
#         super().__init__()
#         self.model=model
#         self.lr=lr
#         self.label_smoothing=label_smoothing
#         self.weight_decay=weight_decay
#         self.lr_decay_rate=lr_decay_rate
#         self.lr_decay_steps=lr_decay_steps
#         self.loss = nn.CrossEntropyLoss(label_smoothing=0.2)

#         if metrics:
#             self.train_metrics = metrics.clone(prefix='train_')
#             self.valid_metrics = metrics.clone(prefix='val_')
#             self.test_metrics = metrics.clone(prefix='test_')
#         else:
#             self.train_metrics=self.valid_metrics=self.test_metrics=None

#     def forward(self,batch):
#         return self.model(batch)
        
#     def training_step(self, batch, _):
#         y = torch.squeeze(batch.pop('label')) # [batch]
#         logits = self.model(batch) # [batch,1,L,W] 
#         logits = F.max_pool2d(logits, kernel_size=logits.size()[2:]) # [batch,1,1,1] 
#         logits = torch.cat( (-logits,logits),axis=1)  # [batch,2,1,1] 
#         logits = torch.squeeze(logits) # [batch,2] for binary classification
#         loss = self.loss(logits, y)
        
#         # Logging..
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         if self.train_metrics:
#             met_out = self.train_metrics(logits[:,1],y)
#             self.log_dict(met_out, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         return loss
    
#     def validation_step(self,batch,_):
#         y,logits,loss=self._shared_eval(batch)
        
#         # Logging..
#         self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         if self.valid_metrics:
#             self._log_metrics(self.valid_metrics,y,logits)
#         return loss
    
#     def test_step(self,batch,_):
#         y,logits,loss=self._shared_eval(batch)
        
#         # Logging..
#         self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
#         if self.test_metrics:
#             self._log_metrics(self.test_metrics,y,logits)
#         return loss

#     def _shared_eval(self,batch):
#         y = torch.squeeze(batch.pop('label')) # [batch]
#         logits = self.model(batch) # [batch,1,L,W] 
#         logits = F.max_pool2d(logits, kernel_size=logits.size()[2:]) # [batch,1,1,1] 
#         logits = torch.cat( (-logits,logits),axis=1)  # [batch,2,1,1] 
#         logits = torch.squeeze(logits) # [batch,2] for binary classification
#         loss = self.loss(logits, y)
#         return y,logits,loss
    
#     def _log_metrics(self,metrics,y,logits):
#         met_out = metrics(logits[:,1],y)
#         self.log_dict(met_out, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    
#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), 
#                                lr=self.lr,
#                                weight_decay=self.weight_decay)
        
#         # Define the learning rate scheduler
#         scheduler = {
#             'scheduler': StepLR(optimizer, 
#                                 step_size=self.lr_decay_steps, 
#                                 gamma=self.lr_decay_rate),
#             'interval': 'epoch',  # Adjust the learning rate at the end of each epoch
#             'frequency': 1  # Apply the scheduler every epoch
#         }
#         return [optimizer], [scheduler]


class TornadoLikelihood(nn.Module):
    """
    Produces 2D tornado likelihood field
    """
    def __init__(self,  shape:Tuple[int]=(2,120,240),
                        c_shape:Tuple[int]=(2,120,240),
                        input_variables:List[str]=ALL_VARIABLES,
                        start_filters:int=64,
                        background_flag:float=-3.0,
                        include_range_folded:bool=True):
        super(TornadoLikelihood, self).__init__()
        self.input_shape=shape
        self.n_sweeps=shape[0]
        self.c_shape=c_shape
        self.input_variables=input_variables
        self.start_filters=start_filters
        self.background_flag=background_flag
        self.include_range_folded=include_range_folded
        
        # Set up normalizers
        self.input_norm_layers = {}
        for v in input_variables:
            min_max = np.array(CHANNEL_MIN_MAX[v]) # [2,]
            scale = 2/(min_max[1]-min_max[0])
            offset = (min_max[0] + min_max[1]) / 2
            self.input_norm_layers[v] = NormalizeVariable(scale,offset)
            
        # Processing blocks
        in_channels = (len(input_variables)+int(self.include_range_folded))*self.n_sweeps
        in_coords=self.c_shape[0]
        self.blk1 = VggBlock(input_image_channels=in_channels, input_coordinate_channels=in_coords, n_output_channels=start_filters, kernel_size=3, n_convs=2, drop_rate=0.1)   # (60,120)
        self.blk2 = VggBlock(input_image_channels=start_filters, input_coordinate_channels=in_coords, n_output_channels=2*start_filters, kernel_size=3, n_convs=2, drop_rate=0.1)  # (30,60)
        self.blk3 = VggBlock(input_image_channels=2*start_filters, input_coordinate_channels=in_coords, n_output_channels=4*start_filters, kernel_size=3, n_convs=3, drop_rate=0.1)  # (15,30)
        self.blk4 = VggBlock(input_image_channels=4*start_filters, input_coordinate_channels=in_coords, n_output_channels=8*start_filters, kernel_size=3, n_convs=3, drop_rate=0.1)  # (7,15)
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=8*start_filters, out_channels=512, kernel_size=(1,1)),
             nn.ReLU(),
             nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1)),
             nn.ReLU()
        )
        self.conv_out = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1,1))
        
        
    def _normalize_inputs(self,data):
        normed_data = {}
        for v in self.input_variables:
            normed_data[v] = self.input_norm_layers[v](data[v])
        return normed_data
    
    def forward(self,data: Dict[str,Any]):
        """
        Assumes data contains radar variables on [batch,tilt,az,rng]
        and coordinate tensor
        """
        # extract inputs
        x = {v:data[v] for v in self.input_variables} # each [batch,tilt,Az,Rng]
        c = data['coordinates']

        
        # normalize
        x = self._normalize_inputs(x) # each [batch,tilt,Az,Rng]
        
        # concatenate along channel (tilt) dim
        x = torch.cat([x[v] for v in self.input_variables],axis=1) #  [batch,tilt*len(input_variables)*2,Az,Rng]
        # Remove nan's
        x = torch.where(torch.isnan(x),self.background_flag,x)
        # concat range_Folded mask
        if self.include_range_folded:
            x = torch.cat((x,data['range_folded_mask']),axis=1)
        
        
        # process
        x,c=self.blk1((x,c))
        x,c=self.blk2((x,c))
        x,c=self.blk3((x,c))
        x,c=self.blk4((x,c))
        x = self.head(x)

        # output single channel heatmap for likelihood field
        x = self.conv_out(x)

        return x


class NormalizeVariable(nn.Module):
    """
    Normalize input tensor as (X-offset)*scale
    """
    def __init__(self, scale, offset):
        super(NormalizeVariable, self).__init__()
        self.register_buffer('scale', torch.tensor(scale))
        self.register_buffer('offset', torch.tensor(offset))

    def forward(self, x):
        return (x - self.offset) * self.scale


class VggBlock(nn.Module):
    """
    Processing block based on VGG19, with coord conv layers
    """
    def __init__(self, input_image_channels,
                       input_coordinate_channels,
                       n_output_channels,
                       kernel_size=3,
                       n_convs=3,
                       drop_rate=0.1):
        super(VggBlock, self).__init__()
        self.input_image_channels=input_image_channels
        self.input_coordinate_channels=input_coordinate_channels
        self.n_output_channels=n_output_channels
        self.kernel_size=kernel_size
        self.n_convs=n_convs
        self.drop_rate=drop_rate

        self.steps = []
        for k in range(n_convs):
            if k==0:
                in_channels=input_image_channels
            else:
                in_channels=n_output_channels
            self.steps.append(CoordConv2D(in_image_channels=in_channels,
                                          in_coord_channels=input_coordinate_channels,
                                          out_channels=n_output_channels, 
                                          kernel_size=kernel_size,
                                          padding='same',
                                          activation='relu'))
        self.conv_block = nn.Sequential(*self.steps)
        self.mx=nn.MaxPool2d(2, stride=2)
        self.mc=nn.MaxPool2d(2, stride=2)
        if drop_rate>0:
            self.drop=nn.Dropout(p=drop_rate)
        else:
            self.drop=None

    def forward(self, inputs):
        x,c=inputs
        x,c=self.conv_block((x,c))
        x=self.mx(x)
        c=self.mc(c)
        if self.drop:
            x=self.drop(x)
        return x,c
        