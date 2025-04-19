"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        
        
        
        
        
        
        
        
        
        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn, # 骨架网络在这儿
            return_nodes={
                "trunk_output.block2": "c3", #第几个block输出 命名为或者说作为 “c3" 
                "trunk_output.block3": "c4", # 左边那些个block 是 regnet里的名字但是 在用于fpn时候一般都主动映射成 c3\c4\c5
                "trunk_output.block4": "c5", # 是 64\160\400 通道
            },
        )
        # trunk_output 是特征字典，传播中backbone是在传递包含所有特征的这个字典
        #  .block 是约定在这个架构 regnet 的命名约定 表示不同特征提取阶段
        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224)) # 参数相当于虚拟输入帮助初始化 
        # 设置batchsize 是一种确定能广播然后避免不设定batchsize或设置1可能出问题的惯例
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        
        
        
        
        
        
        
        
        
        
        
        
        #                           # lateral 理解为横向                                          #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        
        # Replace "pass" statement with your code
        self.fpn_params["lateral_conv3"] = nn.Conv2d (
            in_channels = dummy_out_shapes["c3"].size(1),
            out_channels= out_channels,
            kernel_size= 1,
            stride= 1,               
            padding= 0
        )
        self.fpn_params["lateral_conv4"] = nn.Conv2d (
            in_channels = dummy_out_shapes["c4"].size(1),
            out_channels= out_channels,
            kernel_size= 1,
            stride= 1,               
            padding= 0

        )
        self.fpn_params["lateral_conv5"] = nn.Conv2d (
            in_channels = dummy_out_shapes["c5"].size(1),
            out_channels= out_channels,
            kernel_size= 1,
            stride= 1,               
            padding= 0
        )

        self.fpn_params["output_conv3"] = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1  # 保持尺寸不变
        )













        self.fpn_params["output_conv4"] = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.fpn_params["output_conv5"] = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 对 fpn_params 写好卷积层的setting

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    
    
    
    
    
    
    
    
    
    
    
    # 一些初始理解的误区
    # backbone 里的c3\c4\c5东西就生成好了 
    # [B,64,28,28]\[B,160,14,14]\[B,400,7,7]

    # 这里在做的是 FPN
    # 所以要让维度统一回来 还用stride为1（不改变尺寸） 的 3x3
    # 卷积核来平滑化局部特征 消除从c5到c3上采样过程的不连续 


    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        

        # from deeper/higher layers

        c5 = backbone_feats["c5"]
        p5 = self.fpn_params["lateral_conv5"](c5)
        p5 = self.fpn_params["output_conv5"](p5)
        # first step:  change channels by 1x1 convo filter
        # second step:  change sizes by 3x3 convo filter
        fpn_feats["p5"] = p5
        
        
        c4 = backbone_feats["c4"]
        p4 = self.fpn_params["lateral_conv4"](c4)
        
        
        p4 = p4 + F.interpolate(p5 , size = c4.shape[-2:], mode = "nearest")
        
        # 对于更浅的层 要加入对深层本质特征的融合 同时要对齐尺寸
        # 就必须进行上采样（填充） 这里用的是 nearest 插值

        
        
        
        
        
        
        
        
        
        p4 = self.fpn_params["output_conv4"](p4)
        fpn_feats["p4"] = p4

        c3 = backbone_feats ["c3"]
        p3 = self.fpn_params["lateral_conv3"](c3)
        p3 = p3 + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.fpn_params["output_conv3"](p3)
        fpn_feats ["p3"]  = p3
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    
    
    
    
    
    
    
    
    
    
    
    
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """











    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    
    
    
    
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]
        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        H, W = feat_shape[2] , feat_shape[3]
        y_1 = torch.arange( 0, H , dtype=dtype, device=device)
        x_1 = torch.arange( 0, W, dtype=dtype, device = device)
        # 坐标系形状改成 H*W Row 2 Col ， meshgrid 方法和 stack 方法需要时间理解咯 indexing = 'ij' 是行优先矩阵常规索引模式
        y_x_grid = torch.stack( torch.meshgrid( y_1 , x_1 , indexing='ij') , dim = -1)
        y_x_grid = y_x_grid.reshape( -1 , 2 )
        location_coords[level_name] = (y_x_grid + 0.5) * level_stride # 要求的是上采样扩展回去的中心处
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords




def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    # Sort scores in descending order and get the indices
    _, sorted_indices = scores.sort(descending=True)
    keep = []
    
    while sorted_indices.numel() > 0:
        # Get the index of the highest scoring box
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        














        if sorted_indices.size(0) == 1:
            break
            
        # Get the current box
        current_box = boxes[current_idx].unsqueeze(0)
        
        # Get remaining boxes
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes[remaining_indices]
        # Calculate IoU between current box and remaining boxes
        # Compute intersection coordinates
        x1 = torch.max(current_box[:, 0], remaining_boxes[:, 0])
        y1 = torch.max(current_box[:, 1], remaining_boxes[:, 1])
        x2 = torch.min(current_box[:, 2], remaining_boxes[:, 2])


















        y2 = torch.min(current_box[:, 3], remaining_boxes[:, 3])
        
        # Compute intersection areas
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Compute areas
        current_area = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
        remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        
        # Compute union
        union = current_area + remaining_areas - intersection
        
        # Compute IoU
        iou = intersection / union
        
        # Keep only boxes with IoU <= threshold
        mask = iou <= iou_threshold
        sorted_indices = remaining_indices[mask]
    
    keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    return keep




















def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.
    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
