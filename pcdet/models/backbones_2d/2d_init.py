from .base_bev_backbone import BaseBEVBackbone
from .base_bev_backbone_cspbased import BaseBEVBackbone_cspbased
from .backbone_cspdarknet53 import CSPDarknet53
from .backbone_cspdarknet53_fpn import CSPDarknet53_fpn
from .backbone_cspdarknet53_group import CSPDarknet53_g
from .backbone_cspdarknet53_group_selayer import CSPDarknet53_gs
from .backbone_cspdarknet53_group_selayer_small import CSPDarknet53_gs_small
from .backbone_cspdarknet53_group_selayer_smaller import CSPDarknet53_gs_smaller
from .backbone_mobilenet_v2 import MobileNetV2
from .backbone_resnet50 import ResNet50
from .backbone_sp_res18 import SpRes18
from .backbone_LeapCspNet import LeapCspNet
from .base_bev_backbone_qat import BaseBEVBackbone_qat
from .backbone_cspdarknet53_qat import CSPDarknet53_qat
from .base_bev_backbone_multitask import BaseBEVBackbone_multitask

__all__ = {
    'BaseBEVBackbone': BaseBEVBackbone,
    'BaseBEVBackbone_cspbased':BaseBEVBackbone_cspbased,
    'CSPDarknet53': CSPDarknet53,
    'CSPDarknet53_g': CSPDarknet53_g,
    'CSPDarknet53_gs': CSPDarknet53_gs,
    'CSPDarknet53_gs_small': CSPDarknet53_gs_small,
    'CSPDarknet53_gs_smaller': CSPDarknet53_gs_smaller,
    'MobileNetV2': MobileNetV2,
    'ResNet50': ResNet50,
    'SpRes18': SpRes18,
    'LeapCspNet': LeapCspNet,
    'CSPDarknet53_fpn': CSPDarknet53_fpn,
    'BaseBEVBackbone_qat': BaseBEVBackbone_qat,
    'CSPDarknet53_qat': CSPDarknet53_qat,
    'BaseBEVBackbone_multitask': BaseBEVBackbone_multitask
}
