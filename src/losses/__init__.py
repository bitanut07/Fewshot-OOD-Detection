# Losses
from .classification_loss import ClassificationLoss
from .contrastive_loss import LocalContrastiveLoss
from .alignment_loss import GlobalAlignmentLoss, LocalAlignmentLoss
from .total_loss import TotalLoss
__all__ = ["ClassificationLoss", "LocalContrastiveLoss", "GlobalAlignmentLoss", "LocalAlignmentLoss", "TotalLoss"]
