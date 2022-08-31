import math
from typing import Optional

import torch.nn.functional as F
from torch import Tensor as T, nn


class SRInvariantConv1d:
    def __init__(self,
                 conv: nn.Conv1d,
                 in_sr: Optional[int] = None,
                 out_sr: Optional[int] = None) -> None:
        assert conv.dilation == (1,)  # TODO(cm): remove later
        self.conv = conv
        self.ratio = 1.0
        if in_sr is not None and out_sr is not None:
            assert out_sr >= in_sr
            if in_sr != out_sr:
                self.ratio = out_sr / in_sr
        self.ratio_ceil = math.ceil(self.ratio)

        self.in_sr = in_sr
        self.out_sr = out_sr

    def forward(self, audio: T) -> T:
        if self.ratio == 1.0:
            return self.conv(audio)

        if self.ratio.is_integer():
            out = F.conv1d(audio,
                           self.conv.weight,
                           self.conv.bias,
                           self.conv.stride,
                           self.conv.padding,  # TODO(cm): fix padding
                           dilation=(int(self.ratio),),
                           groups=self.conv.groups)
            return out

        out_hi = F.conv1d(audio,
                          self.conv.weight,
                          self.conv.bias,
                          self.conv.stride,
                          padding="same",  # TODO(cm): fix padding
                          dilation=(self.ratio_ceil,),
                          groups=self.conv.groups)
        out_lo = F.conv1d(audio,
                          self.conv.weight,
                          self.conv.bias,
                          self.conv.stride,
                          padding="same",  # TODO(cm): fix padding
                          dilation=(self.ratio_ceil - 1,),
                          groups=self.conv.groups)

        # TODO(cm): look into fancier interpolation
        alpha = self.ratio_ceil - self.ratio
        out_hi *= (1.0 - alpha)
        out_lo *= alpha
        return out_hi + out_lo
