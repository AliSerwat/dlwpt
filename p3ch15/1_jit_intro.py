# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch
import re
def xprint(s):
    s = str(s)
    s = re.sub(' *#.*','',s)
    print(s)


def myfn(x):
    y = x[0]
    for i in range(1, x.size(0)):
        y = y + x[i]
    return y


inp = torch.randn(5,5)
traced_fn = torch.jit.trace(myfn, inp)
print(traced_fn.code)

scripted_fn = torch.jit.script(myfn)
print(scripted_fn.code)

xprint(scripted_fn.graph)

import sys
sys.path.append('..')

from p2ch13.model_seg import UNetWrapper

# +
seg_dict = torch.load('../data-unversioned/part2/models/p2ch13/seg_2019-10-20_15.57.21_none.best.state', map_location='cpu')
seg_model = UNetWrapper(in_channels=8, n_classes=1, depth=4, wf=3, padding=True, batch_norm=True, up_mode='upconv')
seg_model.load_state_dict(seg_dict['model_state'])
seg_model.eval()
for p in seg_model.parameters():
    p.requires_grad_(False)

traced_seg_model = torch.jit.trace()
# -


