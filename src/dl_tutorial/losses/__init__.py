from __future__ import annotations

import torch.nn as nn

from dl_tutorial.registry import LOSSES


@LOSSES.register("cross_entropy")
class CrossEntropy(nn.CrossEntropyLoss):
    pass


@LOSSES.register("bce_logits")
class BCEWithLogits(nn.BCEWithLogitsLoss):
    pass
