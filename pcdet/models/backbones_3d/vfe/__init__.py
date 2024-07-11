from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE
}

from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_pillar_vfe import DynamicPillarVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate
from .pillar_vfe_TA import PillarVFE_TA
from .pillar_vfe_TA_linear2conv_va import PillarVFE_TA_va
from .pillar_vfe_lane import PillarVFE_TA_va_lane
from .pillar_vfe_TA_linear2conv_paca import PillarVFE_TA_paca
from .pillar_vfe_TA_linear2conv_pacava import PillarVFE_TA_pacava
__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'PillarVFE_TA': PillarVFE_TA,
    'PillarVFE_TA_va': PillarVFE_TA_va,
    'PillarVFE_TA_paca': PillarVFE_TA_paca,
    'PillarVFE_TA_pacava': PillarVFE_TA_pacava,
    'PillarVFE_TA_va_lane': PillarVFE_TA_va_lane
}

