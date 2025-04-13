# models/escnn_dsf.py
import torch
import torch.nn as nn
import escnn
from escnn import gspaces, nn as enn

# Define a DenseBlock that maintains equivariance
class EquivariantDenseBlock(enn.EquivariantModule):
    def __init__(self, in_type, growth_rate, num_layers=4, dropout_rate=0.2):
        """
        Equivariant implementation of a DenseBlock
        
        Args:
            in_type: Input FieldType
            growth_rate: Growth rate (k in DenseNet paper)
            num_layers: Number of layers in the block
            dropout_rate: Dropout rate
        """
        super(EquivariantDenseBlock, self).__init__()
        
        self.in_type = in_type
        self.layers = nn.ModuleList()
        
        # The current input type starts as the block's input type
        curr_in_type = in_type
        
        # Build the layers
        for i in range(num_layers):
            # BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3) pattern from DenseNet
            layer = enn.SequentialModule(
                # 1x1 convolution to reduce channels (bottleneck)
                enn.NormBatchNorm(curr_in_type),
                enn.ReLU(curr_in_type),
                enn.R2Conv(curr_in_type, 
                          enn.FieldType(curr_in_type.gspace, 
                                        growth_rate * 4 * [curr_in_type.gspace.regular_repr]),
                          kernel_size=1, 
                          padding=0,
                          bias=False),
                
                # 3x3 convolution 
                enn.NormBatchNorm(enn.FieldType(curr_in_type.gspace, 
                                               growth_rate * 4 * [curr_in_type.gspace.regular_repr])),
                enn.ReLU(enn.FieldType(curr_in_type.gspace, 
                                      growth_rate * 4 * [curr_in_type.gspace.regular_repr])),
                enn.R2Conv(enn.FieldType(curr_in_type.gspace, 
                                        growth_rate * 4 * [curr_in_type.gspace.regular_repr]),
                          enn.FieldType(curr_in_type.gspace, 
                                        growth_rate * [curr_in_type.gspace.regular_repr]),
                          kernel_size=3, 
                          padding=1,
                          bias=False),
                
                # Dropout for regularization
                enn.PointwiseDropout(enn.FieldType(curr_in_type.gspace, 
                                                  growth_rate * [curr_in_type.gspace.regular_repr]),
                                    p=dropout_rate)
            )
            
            self.layers.append(layer)
            
            # Update the input type for the next layer by concatenating
            # the current input with the new features
            channels_in = curr_in_type.size
            channels_new = growth_rate
            
            # Create a new field type with concatenated channels
            concat_type = enn.FieldType(curr_in_type.gspace,
                                       (channels_in + channels_new) * [curr_in_type.gspace.regular_repr])
            curr_in_type = concat_type
        
        # Store the output type
        self.out_type = curr_in_type
    
    def forward(self, x):
        """
        Forward pass concatenating the output of each layer with all inputs
        
        Args:
            x: Input tensor
        """
        features = x
        
        for i, layer in enumerate(self.layers):
            new_features = layer(features)
            # Concatenate along the channel dimension while preserving equivariance
            features = enn.tensor_directsum([features, new_features])
            
        return features
    
    def evaluate_output_shape(self, input_shape):
        """Calculate the output shape given the input shape"""
        return input_shape  # Spatial dimensions remain the same


# Define a transition layer between dense blocks
class EquivariantTransition(enn.EquivariantModule):
    def __init__(self, in_type, reduction=0.5):
        """
        Transition layer to reduce spatial dimensions and channels
        
        Args:
            in_type: Input FieldType
            reduction: Channel reduction factor
        """
        super(EquivariantTransition, self).__init__()
        
        self.in_type = in_type
        out_channels = int(in_type.size * reduction)
        
        # Create output type with reduced channels
        self.out_type = enn.FieldType(in_type.gspace, 
                                     out_channels * [in_type.gspace.regular_repr])
        
        # BN-Conv(1x1)-AvgPool(2x2)
        self.layers = enn.SequentialModule(
            enn.NormBatchNorm(in_type),
            enn.ReLU(in_type),
            enn.R2Conv(in_type, self.out_type, kernel_size=1, padding=0, bias=False),
            enn.PointwiseAvgPool(self.out_type, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def evaluate_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] // 2, input_shape[2] // 2

# models/escnn_dsf.py (continued)

class ESCNN_DSF(enn.EquivariantModule):
    def __init__(self, num_classes=2, growth_rate=32, block_config=(6, 12, 24, 16),
                 group_order=8, dropout_rate=0.2, compression=0.5):
        """
        DSF-CNN equivalent using escnn
        
        Args:
            num_classes: Number of output classes (2 for PCam)
            growth_rate: Growth rate in dense blocks (k in DenseNet paper)
            block_config: Number of layers in each dense block
            group_order: Order of the rotation group (C8 = 8 rotations)
            dropout_rate: Dropout rate
            compression: Compression factor in transition layers
        """
        super(ESCNN_DSF, self).__init__()
        
        # Setup the group structure (rotation group C8)
        # self.r2_act = gspaces.Rot2dOnR2(group_order) # outdated? 
        self.r2_act = gspaces.rot2dOnR2(group_order)

        # Initial convolutional layer
        # RGB image has 3 channels with trivial representation
        self.in_type = enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])
        
        # Initial convolution (7x7) followed by max pooling
        self.features = enn.SequentialModule(
            # Convolve from trivial RGB to regular representation
            enn.R2Conv(self.in_type, 
                      enn.FieldType(self.r2_act, 
                                   2 * growth_rate * [self.r2_act.regular_repr]),
                      kernel_size=7, 
                      stride=2, 
                      padding=3, 
                      bias=False),
            enn.NormBatchNorm(enn.FieldType(self.r2_act, 
                                           2 * growth_rate * [self.r2_act.regular_repr])),
            enn.ReLU(enn.FieldType(self.r2_act, 
                                  2 * growth_rate * [self.r2_act.regular_repr])),
            enn.PointwiseMaxPool(enn.FieldType(self.r2_act, 
                                              2 * growth_rate * [self.r2_act.regular_repr]),
                                kernel_size=3,
                                stride=2,
                                padding=1)
        )
        
        # Current number of feature maps
        curr_channels = 2 * growth_rate
        
        # Current input type to the next layer
        curr_type = enn.FieldType(self.r2_act, 
                                 curr_channels * [self.r2_act.regular_repr])
        
        # Dense blocks with transitions
        # Each dense block consists of multiple layers that concatenate features
        for i, num_layers in enumerate(block_config):
            # Add a dense block
            block = EquivariantDenseBlock(
                curr_type,
                growth_rate,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            
            # Update current type and channels
            curr_type = block.out_type
            curr_channels = curr_type.size
            
            # Add a transition layer after each dense block except the last one
            if i != len(block_config) - 1:
                transition = EquivariantTransition(curr_type, reduction=compression)
                self.features.add_module(f'transition{i+1}', transition)
                curr_type = transition.out_type
                curr_channels = curr_type.size
        
        # Final batch norm
        self.features.add_module('norm_final', 
                                enn.NormBatchNorm(curr_type))
        self.features.add_module('relu_final',
                                enn.ReLU(curr_type))
        
        # Global average pooling and classification
        # First, pool over spatial dimensions
        self.pooling = enn.PointwiseAdaptiveAvgPool(curr_type, (1, 1))
        
        # Then, pool over group dimension to achieve invariance
        self.invariant_map = enn.GroupPooling(curr_type)
        
        # Get the output type after group pooling
        invariant_type = self.invariant_map.out_type
        
        # Final classification layer
        self.classifier = enn.SequentialModule(
            # Flatten
            enn.Flatten(invariant_type),
            # Fully connected layer
            enn.Linear(invariant_type, 
                      enn.FieldType(self.r2_act, num_classes * [self.r2_act.trivial_repr])),
            # Remove redundant dimensions
            enn.GatherScalar()
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights in accordance with the original DenseNet paper"""
        for name, module in self.named_modules():
            if isinstance(module, enn.R2Conv):
                # He initialization
                enn.init.generalized_he_init(module.kernel, module.basisexpansion)
            elif isinstance(module, enn.Linear):
                # Xavier initialization
                enn.init.generalized_xavier_init(module.weights)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, 3, H, W]
        """
        # Convert the standard tensor to a G-steerable tensor
        x = enn.GeometricTensor(x, self.in_type)
        
        # Feature extraction
        features = self.features(x)
        
        # Global average pooling
        pooled = self.pooling(features)
        
        # Group pooling for rotation invariance
        invariant = self.invariant_map(pooled)
        
        # Classification
        output = self.classifier(invariant)
        
        return output
    
    def evaluate_output_shape(self, input_shape):
        """
        Compute the output shape for a given input shape.

        Args:
            input_shape: Shape of the input tensor (batch_size, channels, height, width)
            
        Returns:
            The shape of the output tensor (batch_size, num_classes)
        """
        # After all operations, we get a tensor with shape (batch_size, num_classes)
        return (input_shape[0], self.classifier[-1].out_type.size)