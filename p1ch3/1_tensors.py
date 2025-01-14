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

a = [1.0, 2.0, 1.0]

a[0]

a[2] = 3.0
a

import torch # <1>
a = torch.ones(3) # <2>
a

a[1]

float(a[1])

a[2] = 2.0
a

points = torch.zeros(6) # <1>
points[0] = 4.0 # <2>
points[1] = 1.0
points[2] = 5.0
points[3] = 3.0
points[4] = 2.0
points[5] = 1.0

points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])
points

float(points[0]), float(points[1])

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points

points.shape

points = torch.zeros(3, 2)
points

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points

points[0, 1]

points[0]

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storage()

points_storage = points.storage()
points_storage[0]

points.storage()[1]

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storage = points.storage()
points_storage[0] = 2.0
points

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point.storage_offset()

second_point.size()

second_point.shape

points.stride()

second_point = points[1]
second_point.size()

second_point.storage_offset()

second_point.stride()

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point[0] = 10.0
points

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1].clone()
second_point[0] = 10.0
points

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points

points_t = points.t()
points_t

id(points.storage()) == id(points_t.storage())

points.stride()

points_t.stride()

some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)
some_t.shape

transpose_t.shape

some_t.stride()

transpose_t.stride()

points.is_contiguous()

points_t.is_contiguous()

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_t = points.t()
points_t

points_t.storage()

points_t.stride()

points_t_cont = points_t.contiguous()
points_t_cont

points_t_cont.stride()

points_t_cont.storage()

double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)

short_points.dtype

double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()

double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10, 2).to(dtype=torch.short)

points_64 = torch.rand(5, dtype=torch.double)  # <1>
points_short = points_64.to(torch.short)
points_64 * points_short  # works from PyTorch 1.3 onwards

# reset points back to original value
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

some_list = list(range(6))
some_list[:]     # <1>
some_list[1:4]   # <2>
some_list[1:]    # <3>
some_list[:4]    # <4>
some_list[:-1]   # <5>
some_list[1:4:2] # <6>

points[1:]       # <1>
points[1:, :]    # <2>
points[1:, 0]    # <3>
points[None]     # <4>

points = torch.ones(3, 4)
points_np = points.numpy()
points_np

points = torch.from_numpy(points_np)

torch.save(points, '../data/p1ch3/ourpoints.t')

with open('../data/p1ch3/ourpoints.t','wb') as f:
   torch.save(points, f)

points = torch.load('../data/p1ch3/ourpoints.t')

with open('../data/p1ch3/ourpoints.t','rb') as f:
   points = torch.load(f)

# +
import h5py

f = h5py.File('../data/p1ch3/ourpoints.hdf5', 'w')
dset = f.create_dataset('coords', data=points.numpy())
f.close()
# -

f = h5py.File('../data/p1ch3/ourpoints.hdf5', 'r')
dset = f['coords']
last_points = dset[-2:]

last_points = torch.from_numpy(dset[-2:])
f.close()

points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')

points_gpu = points.to(device='cuda')

points_gpu = points.to(device='cuda:0')

points = 2 * points  # <1>
points_gpu = 2 * points.to(device='cuda')  # <2>

points_gpu = points_gpu + 4

points_cpu = points_gpu.to(device='cpu')

points_gpu = points.cuda()  # <1>
points_gpu = points.cuda(0)
points_cpu = points_gpu.cpu()

# +
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)

a.shape, a_t.shape

# +
a = torch.ones(3, 2)
a_t = a.transpose(0, 1)

a.shape, a_t.shape
# -

a = torch.ones(3, 2)

a.zero_()
a
