import torch
import torch.nn as nn


# Basic operations used in our model, namely Linear, Conv3d, Conv2d, ConvTranspose3d and ConvTranspose2d
# Linear: Linear transformation, available in all deep learning libraries
# Conv3d and Con2d: Convolution with 2 or 3 dimensions, available in all deep learning libraries
# ConvTranspose3d, ConvTranspose2d: transposed convolution with 2 or 3 dimensions, see Pytorch API or Tensorflow API
#from Your_AI_Library import Linear, Conv3d, Conv2d, ConvTranspose3d, ConvTranspose2d (nn)

# Functions in the networks, namely GeLU, DropOut, DropPath, LayerNorm, and SoftMax
# GeLU: the GeLU activation function, see Pytorch API or Tensorflow API
# DropOut: the dropout function, available in all deep learning libraries
# DropPath: the DropPath function, see the implementation of vision-transformer, see timm pakage of Pytorch
# A possible implementation of DropPath: from timm.models.layers import DropPath
# LayerNorm: the layer normalization function, see Pytorch API or Tensorflow API
# Softmax: softmax function, see Pytorch API or Tensorflow API
#from Your_AI_Library import GeLU, DropOut, LayerNorm, SoftMax (nn)

# Common functions for roll, pad, and crop, depends on the data structure of your software environment
##from Your_AI_Library import roll3D, pad3D, pad2D, Crop3D, Crop2D

# Common functions for reshaping and changing the order of dimensions
# reshape: change the shape of the data with the order unchanged, see Pytorch API or Tensorflow API
# TransposeDimensions: change the order of the dimensions, see Pytorch API or Tensorflow API
#from Your_AI_Library import reshape, TransposeDimensions (torch)

# Common functions for creating new tensors
# ConstructTensor: create a new tensor with an arbitrary shape
# TruncatedNormalInit: Initialize the tensor with Truncate Normalization distribution
# RangeTensor: create a new tensor like range(a, b)
#from Your_AI_Library import ConstructTensor, TruncatedNormalInit, RangeTensor

# Common operations for the data, you may design it or simply use deep learning APIs default operations
# LinearSpace: a tensor version of numpy.linspace
# MeshGrid: a tensor version of numpy.meshgrid
# Stack: a tensor version of numpy.stack
# Flatten: a tensor version of numpy.ndarray.flatten
# TensorSum: a tensor version of numpy.sum
# TensorAbs: a tensor version of numpy.abs
# Concatenate: a tensor version of numpy.concatenate
#from Your_AI_Library import LinearSpace, MeshGrid, Stack, Flatten, TensorSum, TensorAbs, Concatenate

# Common functions for training models
# LoadModel and SaveModel: Load and save the model, some APIs may require further adaptation to hardwares
# Backward: Gradient backward to calculate the gratitude of each parameters
# UpdateModelParametersWithAdam: Use Adam to update parameters, e.g., torch.optim.Adam
#from Your_AI_Library import LoadModel, Backward, UpdateModelParametersWithAdam, SaveModel

from timm.models.layers import DropPath, trunc_normal_
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import reduce, lru_cache
from operator import mul
import torch.nn.functional as F


class PanguModel(nn.Module):
  def __init__(self):
    super(PanguModel, self).__init__()
    # Drop path rate is linearly increased as the depth increases
    drop_path_list = torch.linspace(0, 0.2, 8)

    # Patch embedding
    self._input_layer = PatchEmbedding((5, 4),(4, 4, 2), 2)#batchsize = 2

    # Four basic layers
    self.layer1 = EarthSpecificLayer(2, 2, drop_path_list[:2], 1, Z = 8, H = 360, W = 180)#batchsize = 2
    self.layer2 = EarthSpecificLayer(6, 4, drop_path_list[2:], 2, Z = 8, H = 180, W = 90)#batchsize = 4
    self.layer3 = EarthSpecificLayer(6, 4, drop_path_list[2:], 2, Z = 8, H = 180, W = 90)
    self.layer4 = EarthSpecificLayer(2, 2, drop_path_list[:2], 1, Z = 8, H = 360, W = 180)

    # Upsample and downsample
    self.upsample = UpSample(4, 2)
    self.downsample = DownSample(2)

    # Patch Recovery
    self._output_layer = PatchRecovery(4, (4, 4, 2))
    
  def forward(self, input, input_surface):
    '''Backbone architecture'''
    # Embed the input fields into patches
    print('input.size: ', input.shape)
    print('input_surface.size: ', input_surface.shape)
    
    patch=[4,4,2]
    
    x = self._input_layer(input, input_surface)
    # Encoder, composed of two layers
    # Layer 1, shape (8, 360, 181, C), C = 192 as in the original paper
    x = self.layer1(x, 8, 360, 180)

    # Store the tensor for skip-connection
    skip = x

    # Downsample from (8, 360, 181) to (8, 180, 91)
    x = self.downsample(x, 8, 360, 180)

    # Layer 2, shape (8, 180, 91, 2C), C = 192 as in the original paper
    x = self.layer2(x, 8, 180, 90)

    # Decoder, composed of two layers
    # Layer 3, shape (8, 180, 91, 2C), C = 192 as in the original paper
    x = self.layer3(x, 8, 180, 90) 

    # Upsample from (8, 180, 91) to (8, 360, 181)
    x = self.upsample(x, 8, 180, 90)

    # Layer 4, shape (8, 360, 181, 2C), C = 192 as in the original paper
    x = self.layer4(x, 8, 360, 180) 

    # Skip connect, in last dimension(C from 192 to 384)
    x = torch.cat((skip, x), axis=-1)

    # Recover the output fields from patches
    output, output_surface = self._output_layer(x, 8, 360, 180)
    return output, output_surface

class PatchEmbedding(nn.Module):
  def __init__(self, input_dim, patch_size, dim):
    super(PatchEmbedding, self).__init__()
    '''Patch embedding operation'''
    # Here we use convolution to partition data into cubes
    self.conv = nn.Conv3d(in_channels=input_dim[0], out_channels=dim, kernel_size=patch_size, stride=patch_size)
    self.conv_surface = nn.Conv2d(in_channels=input_dim[1], out_channels=dim, kernel_size=patch_size[:2], stride=patch_size[:2])
    self.patch = patch_size
    # Load constant masks from the disc
    ##self.land_mask, self.soil_type, self.topography = LoadConstantMask()
    
  def forward(self, input, input_surface):
    # Zero-pad the input
    #m =  nn.ConstantPad3d((0,1,0,0,0,0), 0)#left,right,top,down,front,back 4, 4, 3, 3, 2, 2, 1, 1
    #input = m(input)
    input = F.pad(input, pad=(0, 0, 0, 0, 0, 0, 0, 1))
    #print('padinput: ',input.shape)#[192, 13, 1440, 721, 5] [192, 14, 1440, 724, 5]
    #n = nn.ZeroPad2d((0,0,3,0))#left,right,top,down
    #input_surface = n(input_surface)
    #input_surface = F.pad(input_surface, pad=(0, 0, 0, 3, 0, 0))
    #print('padinput_surface: ',input_surface.shape)#[192, 1440, 721, 4] [192, 1440, 724, 4]
    
    input = input.permute(0,4,2,3,1)
    input_surface = input_surface.permute(0,3,1,2)
    print('transinput: ',input.shape)
    print('transinput_surface: ',input_surface.shape)
    
    # Apply a linear projection for patch_size[0]*patch_size[1]*patch_size[2] patches, patch_size = (2, 4, 4) as in the original paper
    input = self.conv(input)
    print('convinput: ',input.shape)#torch.Size([192, 192, 721, 180, 1])

    # Add three constant fields to the surface fields
    #input_surface =  torch.cat(input_surface, self.land_mask, self.soil_type, self.topography)

    # Apply a linear projection for patch_size[1]*patch_size[2] patches
    input_surface = self.conv_surface(input_surface)
    print('convinput_surface: ',input_surface.shape)
    
    input_surface = torch.unsqueeze(input_surface, -1)
    print('unsqinput_surface: ',input_surface.shape)
    # Concatenate the input in the pressure level, i.e., in Z dimension
    x = torch.cat((input, input_surface),-1)
    print('catx: ',x.shape)

    # Reshape x for calculation of linear projections
    x = x.permute(0, 2, 3, 4, 1)
    print('transx: ',x.shape)
    x = torch.reshape(x, (x.shape[0], 8*360*180, x.shape[-1]))
    print('Embx: ',x.shape)
    return x
    
class PatchRecovery(nn.Module):
  def __init__(self, dim, patch_size):
    super(PatchRecovery, self).__init__()
    '''Patch recovery operation'''
    # Hear we use two transposed convolutions to recover data
    self.patch_size = patch_size
    self.conv = nn.ConvTranspose3d(in_channels=dim, out_channels=5, kernel_size=patch_size, stride=patch_size)
    self.conv_surface = nn.ConvTranspose2d(in_channels=dim, out_channels=4, kernel_size=patch_size[:2], stride=patch_size[:2])
    
  def forward(self, x, Z, H, W): #!!!!!!!!!!!!!!!
    # The inverse operation of the patch embedding operation, patch_size = (2, 4, 4) as in the original paper
    # Reshape x back to three dimensions
    x = x.permute(0, 2, 1)
    print('x_per: ', x.shape)
    x = x.reshape(x.shape[0], x.shape[1], Z, H, W) #!!!!!!!!!!!!!!!
    print('x_re: ', x.shape)#[2, 4, 8, 360, 180]
    x = x.permute(0, 1, 3, 4, 2)
    # Call the transposed convolution
    output = self.conv(x[:, :, :, :, 1:])
    print('output_x: ',output.shape)
    output_surface = self.conv_surface(x[:, :, :, :, 0])

    # Crop the output to remove zero-paddings
    #output = Crop3D(output)
    
    #outputP = nn.ConstantPad3d((0,0,-1,0,-1,-2), 0)#left,right,top,down,front,back
    #output = outputP(output)
    output = F.pad(output, pad=(0, -1, 0, 0, 0, 0, 0, 0))
    #output_surface = Crop2D(output_surface)
    #output_surfaceP = nn.ZeroPad2d((0,0,-3,0))#left,right,top,down
    #output_surface = outputP(output_surface)
    #output_surface = F.pad(output_surface, pad=(0, 0, 0, -3, 0, 0))
    return output, output_surface

class DownSample(nn.Module):
  def __init__(self, dim):
    super(DownSample, self).__init__()
    '''Down-sampling operation'''
    # A linear function and a layer normalization
    self.linear = nn.Linear(4*dim, 2*dim, bias=False)
    self.norm = nn.LayerNorm(4*dim)
  
  def forward(self, x, Z, H, W):# !!!!!!!!!!!!!!!
    # Reshape x to three dimensions for downsampling
    print('before_x: ',x.shape)
    x = x.reshape(x.shape[0], Z, H, W, x.shape[-1])#!!!!!!!!!!!!!!!
    print('x_re: ',x.shape)#x_re:  torch.Size([2, 8, 360, 181, 2])
    # Padding the input to facilitate downsampling
    #x = F.pad(x, pad=(0, 0, 0, 1, 0, 0, 0, 0))
    #print('pad_x: ',x.shape)
    # Reorganize x to reduce the resolution: simply change the order and downsample from (8, 360, 182) to (8, 180, 91)
    # Reshape x to facilitate downsampling
    Z, H, W = x.shape[1],x.shape[2],x.shape[3]
    x = x.reshape(x.shape[0], Z, H//2, 2, W//2, 2, x.shape[-1])
    print('x_re: ',x.shape)
    # Change the order of x
    x = x.permute(0,1,2,4,3,5,6)
    print('x_per: ',x.shape)
    # Reshape to get a tensor of resolution (8, 180, 91)
    x = x.reshape(x.shape[0], Z*(H//2)*(W//2), 4 * x.shape[-1])
    print('x_re: ',x.shape)
    # Call the layer normalization
    x = self.norm(x)
    print('x_norm: ',x.shape)
    # Decrease the channels of the data to reduce computation cost
    x = self.linear(x)
    print('x_linear: ',x.shape)
    return x

class UpSample(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(UpSample, self).__init__()
    '''Up-sampling operation'''
    # Linear layers without bias to increase channels of the data
    self.linear1 = nn.Linear(input_dim, output_dim*4, bias=False)

    # Linear layers without bias to mix the data up
    self.linear2 = nn.Linear(output_dim, output_dim, bias=False)

    # Normalization
    self.norm = nn.LayerNorm(output_dim)
  
  def forward(self, x, Z, H, W):
    # Call the linear functions to increase channels of the data
    x = self.linear1(x)
    print('x_linear: ',x.shape)
    # Reorganize x to increase the resolution: simply change the order and upsample from (8, 180, 91) to (8, 360, 182)
    # Reshape x to facilitate upsampling.
    
    x = x.reshape(x.shape[0], Z, H, W, 2, 2, x.shape[-1]//4)
    # Change the order of x
    x = x.permute(0,1,2,4,3,5,6)
    print('x_per: ',x.shape)
    # Reshape to get Tensor with a resolution of (8, 360, 182)
    x = x.reshape(x.shape[0], Z, H*2, W*2, x.shape[-1])

    # Crop the output to the input shape of the network
    #x = Crop3D(x)
    #x = functional.center_crop(x,x.size)
    # Reshape x back
    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[-1])

    # Call the layer normalization
    x = self.norm(x)

    # Mixup normalized tensors
    x = self.linear2(x)
    return x
  

class EarthSpecificLayer(nn.Module):
  def __init__(self, depth, dim, drop_path_ratio_list, heads, Z, H, W):
    super().__init__()
    '''Basic layer of our network, contains 2 or 6 blocks'''
    self.depth = depth
    self.blocks = []
    
    # Construct basic blocks
    for i in range(depth):
      self.blocks.append(EarthSpecificBlock(dim, drop_path_ratio_list[i], heads, Z = Z, H = H, W = W))
      
  def forward(self, x, Z, H, W):
    for i in range(self.depth):
      # Roll the input every two blocks
      print(x.shape)
      B, C, D, H, W = x.shape[0], x.shape[-1], Z, H, W
      if i % 2 == 0:
        self.blocks[i](x, Z, H, W, roll=False)
      else:
        self.blocks[i](x, Z, H, W, roll=True)
    return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, D, H, W, C = x.shape[0],x.shape[1],x.shape[2],x.shape[3],x.shape[4]
    print(x.shape)
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    print('x_p:',x.shape)
    #windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    #B = int(windows.shape[0] / (D * H * W / window_size / window_size / window_size))
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x



class EarthSpecificBlock(nn.Module):
  def __init__(self, dim, drop_path_ratio, heads, Z, H, W):
    super().__init__()
    '''
    3D transformer block with Earth-Specific bias and window attention, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    The major difference is that we expand the dimensions to 3 and replace the relative position bias with Earth-Specific bias.
    '''
    # Define the window size of the neural network 
    self.window_size = (2, 6, 10)

    # Initialize serveral operations
    self.drop_path = DropPath(drop_prob=drop_path_ratio)
    self.norm1 = nn.LayerNorm(dim, device='cuda:0')
    self.norm2 = nn.LayerNorm(dim, device='cuda:0')
    self.linear = Mlp(dim, 0)
    self.Z = Z
    self.H = H
    self.W = W
    self.dim = dim
    self.drop_path_ratio = drop_path_ratio
    self.heads = heads
    self.shift_size = tuple(i // 2 for i in self.window_size)
    
    #input_shape=((self.Z * self.H * self.W) // (self.window_size[0] * self.window_size[1] * self.window_size[2]))
    #print('input_shape: ',input_shape)
    
    #self.attention = EarthAttention3D(dim, heads, 0, self.window_size,input_shape=input_shape)

  def forward(self, x, Z, H, W, roll):
    
    
    # Save the shortcut for skip-connection
    shortcut = x

    # Reshape input to three dimensions to calculate window attention
    x = torch.reshape(x, (x.shape[0], Z, H, W, x.shape[2]))
    print('EarthSpecificBlock_x: ',x.shape)#torch.Size([192, 8, 360, 181, 192]
    # Zero-pad input if needed
    #x = F.pad(x, pad=(0, 0, 0, 11, 0, 0))
    #print('padEarthSpecificBlock_x: ',x.shape)#torch.Size([192, 8, 360, 192, 192]
    #x = nn.ZeroPad3d(x)

    # Store the shape of the input for restoration
    ori_shape = x.shape

    if roll:
      # Roll x for half of the window for 3 dimensions
      shifted_x = x.roll(shifts=[self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2], dims=(1, 2, 3))
      # Generate mask of attention masks
      # If two pixels are not adjacent, then mask the attention between them
      # Your can set the matrix element to -1000 when it is not adjacent, then add it to the attention
      
      B, D, H, W, C = x.shape
      img_mask = torch.zeros((1, D, H, W, 1), device='cuda:0')  # 1 H W 1
      h_slices = (slice(0, -self.window_size[0]),
                  slice(-self.window_size[0], -self.shift_size[0]),
                  slice(-self.shift_size[0], None))
      w_slices = (slice(0, -self.window_size[1]),
                  slice(-self.window_size[1], -self.shift_size[1]),
                  slice(-self.shift_size[1], None))
      d_slices = (slice(0, -self.window_size[2]),
                  slice(-self.window_size[2], -self.shift_size[2]),
                  slice(-self.shift_size[2], None))
      cnt = 0
      for d in d_slices:
          for h in h_slices:
              for w in w_slices:
                  img_mask[:, d, h, w, :] = cnt
                  cnt += 1

      mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
      print('mask_windows: ',mask_windows.shape)
      #mask_windows = mask_windows.view(-1, self.window_size * self.window_size * self.window_size)
      mask_windows = mask_windows.squeeze(-1)
      mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
      mask = mask.masked_fill(mask != 0, float(-1000.0)).masked_fill(mask == 0, float(0.0))
      
      #mask = gen_mask(x)
    else:
      # e.g., zero matrix when you add mask to attention
      shifted_x = x
      mask = None

    # Reorganize data to calculate window attention
    #x_window = torch.reshape(x, (x.shape[0], x.shape[1]//self.window_size[0], self.window_size[0], x.shape[2] // self.window_size[1], self.window_size[1], x.shape[3] // self.window_size[2], self.window_size[2], x.shape[-1]))
    #print('EarthSpecificBlock_x_window: ',x_window.shape)
    #x_window = x_window.permute(0, 1, 3, 5, 2, 4, 6, 7)
    #print('transEarthSpecificBlock_x_window: ',x_window.shape)

    # Get data stacked in 3D cubes, which will further be used to calculated attention among each cube
    #x_window = torch.reshape(x_window, (-1, self.window_size[0]* self.window_size[1]*self.window_size[2], x.shape[-1]))
    
    x_window = window_partition(shifted_x, self.window_size)
    print('x_windows_window_partition: ',x_window.shape)#[737280, 144, 192]
    # Apply 3D window attention with Earth-Specific bias
    input_shape = x_window.shape[0]
    self.attention = EarthAttention3D(self.dim, self.heads, 0, self.window_size,input_shape=input_shape)
    attn_windows = self.attention(x_window, mask)
    print('attn_windows: ',x_window.shape)

    # Reorganize data to original shapes
    #x = torch.reshape(x_window, (-1, Z // self.window_size[0], H // self.window_size[1], W // self.window_size[2], self.window_size[0], self.window_size[1], self.window_size[2], self.window_size[-1]))
    #x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)

    # Reshape the tensor back to its original shape
    #x = torch.reshape(x_window, ori_shape)

    x = window_reverse(attn_windows, self.window_size, ori_shape[0], ori_shape[1], ori_shape[2], ori_shape[3])  # B H' W' C
    print('x_window_reverse: ',x.shape)
      
    if roll:
        # Roll x back for half of the window
        x = x.roll(shifts=[self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2], dims=(1, 2, 3))
    else:
        x = x

    # Crop the zero-padding
    #x = F.pad(x, pad=(0, 0, 0, -11, 0, 0))
    #print('unpad_x: ',x.shape)

    # Reshape the tensor back to the input shape
    x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[4]))
    print('x_re: ',x.shape)#[2, 4, 8, 360, 180]
    # Main calculation stages
    x = shortcut + self.drop_path(self.norm1(x))
    x = x + self.drop_path(self.norm2(self.linear(x)))
    print('EarthSpecificBlock_x:', x.shape)#[2, 8, 360, 180, 2]
    return x
    
class EarthAttention3D(nn.Module):
  def __init__(self, dim, heads, dropout_rate, window_size, input_shape):
    super(EarthAttention3D, self).__init__()
    '''
    3D window attention with the Earth-Specific bias, 
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    '''
    # Initialize several operations
    self.linear1 = nn.Linear(dim, dim*3, bias=True, device='cuda:0')
    self.linear2 = nn.Linear(dim, dim, device='cuda:0')
    #self.softmax = torch.softmax(dim=-1)
    self.dropout = nn.Dropout(dropout_rate)

    # Store several attributes
    self.head_number = heads
    self.dim = dim
    self.scale = (dim//heads)**-0.5
    self.window_size = window_size

    # input_shape is current shape of the self.forward function
    # You can run your code to record it, modify the code and rerun it
    # Record the number of different window types
    
    self.type_of_windows = input_shape
    #self.type_of_windows = (Z//window_size[0])*(H//window_size[1])*(W//window_size[2])

    # For each type of window, we will construct a set of parameters according to the paper
    self.earth_specific_bias = nn.Parameter(torch.Tensor((2 * self.window_size[2] - 1) * self.window_size[1] * self.window_size[1] * self.window_size[0] * self.window_size[0], self.type_of_windows, self.head_number))

    # Making these tensors to be learnable parameters
    #self.earth_specific_bias = nn.Parameter(self.earth_specific_bias)

    # Initialize the tensors using Truncated normal distribution
    trunc_normal_(self.earth_specific_bias, std=0.02) 

    # Construct position index to reuse self.earth_specific_bias
    self._construct_index()
    
  def _construct_index(self):
    ''' This function construct the position index to reuse symmetrical parameters of the position bias'''
    # Index in the pressure level of query matrix
    coords_zi = torch.arange(self.window_size[0])
    # Index in the pressure level of key matrix
    coords_zj = -torch.arange(self.window_size[0])*self.window_size[0]

    # Index in the latitude of query matrix
    coords_hi = torch.arange(self.window_size[1])
    # Index in the latitude of key matrix
    coords_hj = -torch.arange(self.window_size[1])*self.window_size[1]

    # Index in the longitude of the key-value pair
    coords_w = torch.arange(self.window_size[2])

    # Change the order of the index to calculate the index in total
    coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
    coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    coords_flatten_1 = torch.flatten(coords_1, start_dim=1) 
    coords_flatten_2 = torch.flatten(coords_2, start_dim=1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0)

    # Shift the index for each dimension to start from 0
    coords[:, :, 2] += self.window_size[2] - 1
    coords[:, :, 1] *= 2 * self.window_size[2] - 1
    coords[:, :, 0] *= (2 * self.window_size[2] - 1)*self.window_size[1]*self.window_size[1]

    # Sum up the indexes in three dimensions
    self.position_index = torch.sum(coords, dim=-1)

    # Flatten the position index to facilitate further indexing
    self.position_index = torch.flatten(self.position_index)
    print('self.position_index: ', self.position_index.shape)
    
  def forward(self, x, mask):
    
    B, N, C = x.shape
    print('x:',x.shape)
    # Linear layer to create query, key and value
    x = self.linear1(x)
    print('x:',x.shape)

    # Record the original shape of the input
    original_shape = x.shape

    # reshape the data to calculate multi-head attention
    qkv = x.view(x.shape[0], x.shape[1], 3, self.head_number, self.dim // self.head_number).permute(2, 0, 3, 1, 4)
    print('qkv: ',qkv.shape)
    #qkv = qkv.permute(2, 0, 3, 1, 4)
    #print('qkv_per: ',qkv.shape)
    query, key, value = qkv[0], qkv[1], qkv[2]
    print('query, key, value: ', query.shape, key.shape, value.shape)

    # Scale the attention
    query = query * self.scale
    print('query: ', query.shape)

    # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
    #attention = query @ key.T # @ denotes matrix multiplication
    #attention = (query @ key.transpose(-2, -1))
    attention = torch.matmul(query, key.transpose(-2, -1))
    print('attention: ',attention.shape)

    # self.earth_specific_bias is a set of neural network parameters to optimize. 
    EarthSpecificBias = self.earth_specific_bias[self.position_index]
    print('EarthSpecificBias: ', EarthSpecificBias.shape)
    # Reshape the learnable bias to the same shape as the attention matrix
    EarthSpecificBias = torch.reshape(EarthSpecificBias, (self.window_size[0]*self.window_size[1]*self.window_size[2], self.window_size[0]*self.window_size[1]*self.window_size[2], self.type_of_windows, self.head_number))
    print('EarthSpecificBias_re: ', EarthSpecificBias.shape)
    EarthSpecificBias = EarthSpecificBias.permute(2, 3, 0, 1)
    print('EarthSpecificBias_permute: ', EarthSpecificBias.shape)
    EarthSpecificBias = torch.reshape(EarthSpecificBias, ([1]+list(EarthSpecificBias.shape)))
    print('EarthSpecificBias_re2: ', EarthSpecificBias.shape)

    # Add the Earth-Specific bias to the attention matrix
    attention = attention.to(torch.device("cuda:0")) + EarthSpecificBias.to(torch.device("cuda:0"))
    print('attention+EarthSpecificBias: ',attention.shape)

    # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
    #attention = self.mask_attention(attention, mask)
    #attention = self.softmax(attention)
    #attention = self.dropout(attention)
    
    if mask is not None:
        nW = mask.shape[0]
        attention = attention.view(B // nW, nW, self.head_number, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attention = attention.view(-1, self.head_number, N, N)
        attention = torch.softmax(attention, dim=-1)
    else:
        attention = torch.softmax(attention, dim=-1)

    attention = self.dropout(attention)
    print('attention_drop: ',attention.shape)
    # Calculated the tensor after spatial mixing.
    #x = attention @ value.T # @ denote matrix multiplication
    x = torch.matmul(attention, value)
    print('x+a: ',x.shape)
    # Reshape tensor to the original shape
    #x = x.permute(0, 2, 1)
    #print('x+a: ',x.shape)
    x = torch.reshape(x, (B, N, C))
    print('x_re: ',x.shape)

    # Linear layer to post-process operated tensor
    x = self.linear2(x)
    x = self.dropout(x)
    return x
  
class Mlp(nn.Module):
  def __init__(self, dim, dropout_rate):
    super(Mlp, self).__init__()
    '''MLP layers, same as most vision transformer architectures.'''
    self.linear1 = nn.Linear(dim, dim * 4, device='cuda:0')
    self.linear2 = nn.Linear(dim * 4, dim, device='cuda:0')
    self.drop = nn.Dropout(dropout_rate)
    
  def forward(self, x):
    x = self.linear1(x)
    x = F.gelu(x)
    x = self.drop(x)
    x = self.linear2(x)
    x = self.drop(x)
    return x
 
#############################################   
device = torch.device("cuda:0")
Pmodel = PanguModel()
Pmodel.to(device)
Pmodel.load_state_dict(torch.load('/home/p131/pangu/save_test.pt'))
Pmodel.eval()

input_data = torch.rand(8, 13, 1440, 720, 5)
input_surface_data = torch.rand(8, 1440, 720, 4)

predict, predict_surface= Pmodel(input_data.to(device),input_surface_data.to(device))
print(predict.shape)
print(predict_surface.shape)
