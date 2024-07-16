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


torch.version.__version__

a = torch.ones(3,3)

b = torch.ones(3,3)

a + b

a = a.to('cuda')
b = b.to('cuda')
a + b


