import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

# from open_clip import CLIP3D
import open_clip

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if len(x.shape) == 4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            elif len(x.shape) == 5:
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
        
        
class Block3D(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., inflate_len=3, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=(inflate_len,7,7), padding=(inflate_len // 2,3,3), groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
class BlockV2(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
class BlockV23D(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0., inflate_len=3,):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=(inflate_len,7,7), padding=(inflate_len // 2,3,3), groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[BlockV2(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model
    
class ConvNeXt3D(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000,
                 inflate_strategy='131',
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(2,4,4), stride=(2,4,4)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=(1,2,2), stride=(1,2,2)),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block3D(dim=dims[i], inflate_len=int(inflate_strategy[j%len(inflate_strategy)]),
                          drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

        self.apply(self._init_weights)
        
    def inflate_weights(self, s_state_dict):
        t_state_dict = self.state_dict()
        from collections import OrderedDict
        for key in t_state_dict.keys():
            if key not in s_state_dict:
                print(key)
                continue
            if t_state_dict[key].shape != s_state_dict[key].shape:
                t = t_state_dict[key].shape[2]
                s_state_dict[key] = s_state_dict[key].unsqueeze(2).repeat(1,1,t,1,1) / t
        self.load_state_dict(s_state_dict, strict=False)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, return_spatial=False, multi=False, layer=-1):
        if multi:
            xs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if multi:
                xs.append(x)
        if return_spatial:
            if multi:
                shape = xs[-1].shape[2:]
                return torch.cat([F.interpolate(x,size=shape, mode="trilinear") for x in xs[:-1]], 1) #+ [self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)], 1)
            elif layer > -1:
                return xs[layer]
            else:
                return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return self.norm(x.mean([-3, -2, -1])) # global average pooling, (N, C, T, H, W) -> (N, C)

    def forward(self, x, multi=False, layer=-1):
        x = self.forward_features(x, True, multi=multi, layer=layer)
        return x


class ConvNeXtV23D(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 inflate_strategy='131',
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=(2,4,4), stride=(2,4,4)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=(1,2,2), stride=(1,2,2)),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[BlockV23D(dim=dims[i], drop_path=dp_rates[cur + j],
                            inflate_len=int(inflate_strategy[j%len(inflate_strategy)]),
                           ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
    
    def inflate_weights(self, pretrained_path):
        t_state_dict = self.state_dict()
        s_state_dict = torch.load(pretrained_path)["model"]
        from collections import OrderedDict
        for key in t_state_dict.keys():
            if key not in s_state_dict:
                print(key)
                continue
            if t_state_dict[key].shape != s_state_dict[key].shape:
                print(t_state_dict[key].shape, s_state_dict[key].shape)
                t = t_state_dict[key].shape[2]
                s_state_dict[key] = s_state_dict[key].unsqueeze(2).repeat(1,1,t,1,1) / t
        self.load_state_dict(s_state_dict, strict=False)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, return_spatial=False, multi=False, layer=-1):
        if multi:
            xs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if multi:
                xs.append(x)
        if return_spatial:
            if multi:
                shape = xs[-1].shape[2:]
                return torch.cat([F.interpolate(x,size=shape, mode="trilinear") for x in xs[:-1]], 1) #+ [self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)], 1)
            elif layer > -1:
                return xs[layer]
            else:
                return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return self.norm(x.mean([-3, -2, -1])) # global average pooling, (N, C, T, H, W) -> (N, C)

    def forward(self, x, multi=False, layer=-1):
        x = self.forward_features(x, True, multi=multi, layer=layer)
        return x


# model_urls = {
#     "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
#     "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
#     "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
#     "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
#     "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
#     "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
#     "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
#     "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
#     "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
# }

# def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
#         checkpoint = torch.hub./apdcephfs/share_1290939/raphaelliu/Vid_Eval/results_LR_256/(url=url, map_location="cpu", check_hash=True)
#         model.load_state_dict(checkpoint["model"])
#     return model

# def convnext_small(pretrained=False,in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
#         checkpoint = torch.hub./apdcephfs/share_1290939/raphaelliu/Vid_Eval/results_LR_256/(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["model"])
#     return model

# def convnext_base(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
#         checkpoint = torch.hub./apdcephfs/share_1290939/raphaelliu/Vid_Eval/results_LR_256/(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["model"])
#     return model


# def convnext_large(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
#         checkpoint = torch.hub./apdcephfs/share_1290939/raphaelliu/Vid_Eval/results_LR_256/(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["model"])
#     return model

# def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
#     if pretrained:
#         assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
#         url = model_urls['convnext_xlarge_22k']
#         checkpoint = torch.hub./apdcephfs/share_1290939/raphaelliu/Vid_Eval/results_LR_256/(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["model"])
        
#     return model

# def convnext_3d_tiny(pretrained=False, in_22k=False, **kwargs):
#     print("Using Imagenet 22K pretrain", in_22k)
#     model = ConvNeXt3D(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
#         checkpoint = torch.hub./apdcephfs/share_1290939/raphaelliu/Vid_Eval/results_LR_256/(url=url, map_location="cpu", check_hash=True)
#         model.inflate_weights(checkpoint["model"])
#     return model

# def convnext_3d_small(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXt3D(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
#     if pretrained:
#         url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
#         checkpoint = torch.hub./apdcephfs/share_1290939/raphaelliu/Vid_Eval/results_LR_256/(url=url, map_location="cpu", check_hash=True)
#         model.inflate_weights(checkpoint["model"])
        
#     return model

model_urls = {
    "convnext_tiny_1k": "../../checkpoints/Dover/convnext_tiny_1k.pth",
    "convnext_small_1k": "../../checkpoints/Dover/convnext_small_1k.pth",
    "convnext_base_1k": "../../checkpoints/Dover/convnext_base_1k.pth",
    "convnext_large_1k": "../../checkpoints/Dover/convnext_large_1k.pth",
    "convnext_tiny_22k": "../../checkpoints/Dover/convnext_tiny_22k.pth",
    "convnext_small_22k": "../../checkpoints/Dover/convnext_small_22k.pth",
    "convnext_base_22k": "../../checkpoints/Dover/convnext_base_22k.pth",
    "convnext_large_22k": "../../checkpoints/Dover/convnext_large_22k.pth",
    "convnext_xlarge_22k": "../../checkpoints/Dover/convnext_xlarge_22k.pth",
}

def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        local_path = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.load(local_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        local_path = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.load(local_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        local_path = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.load(local_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        local_path = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.load(local_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        local_path = model_urls['convnext_xlarge_22k']
        checkpoint = torch.load(local_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

def convnext_3d_tiny(pretrained=False, in_22k=False, **kwargs):
    print("Using Imagenet 22K pretrain", in_22k)
    model = ConvNeXt3D(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        local_path = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.load(local_path, map_location="cpu")
        model.inflate_weights(checkpoint["model"])
    return model

def convnext_3d_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt3D(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        local_path = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.load(local_path, map_location="cpu")
        model.inflate_weights(checkpoint["model"])
    return model

def convnextv2_3d_atto(**kwargs):
    model = ConvNeXtV23D(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    
    return model

def convnextv2_3d_femto(pretrained="../pretrained/convnextv2_femto_1k_224_ema.pt", **kwargs):
    model = ConvNeXtV23D(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    #model.inflate_weights(pretrained)
    return model

def convnextv2_3d_pico(pretrained="../pretrained/convnextv2_pico_1k_224_ema.pt", **kwargs):
    model = ConvNeXtV23D(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    #model.inflate_weights(pretrained)
    return model

def convnextv2_3d_nano(pretrained="../pretrained/convnextv2_nano_1k_224_ema.pt", **kwargs):
    model = ConvNeXtV23D(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    #model.inflate_weights(pretrained)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV23D(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV23D(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV23D(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


    
        
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = convnext_3d_tiny(True).to(device)
    print(model)
    from thop import profile
    print(profile(model, (torch.randn(4,3,32,224,224).to(device),))[0] / 1e9)
    
    