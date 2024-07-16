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

# %matplotlib inline
import copy
import numpy as np
import matplotlib.pyplot as plt

# +
import torch
from p2ch13.dsets import getCandidateInfoList, getCt, old_build2dLungMask
from p2ch13.model_seg import SegmentationMask, MaskTuple
from p2ch13.vis import build2dLungMask
from util.util import xyz2irc


candidateInfo_list = getCandidateInfoList(requireOnDisk_bool=False)
candidateInfo_list[0]
# -

series_list = sorted(set(t.series_uid for t in candidateInfo_list))


# +
def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = copy.deepcopy(cmap)
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.75, N+4)
    return mycmap
tgray = transparent_cmap(plt.cm.gray)
tpurp = transparent_cmap(plt.cm.Purples)
tblue = transparent_cmap(plt.cm.Blues)
tgreen = transparent_cmap(plt.cm.Greens)
torange = transparent_cmap(plt.cm.Oranges)
tred = transparent_cmap(plt.cm.Reds)


clim=(0, 1.3)
start_ndx = 3
mask_model = SegmentationMask().to('cuda')


# +
ct_list = []
for nit_ndx in range(start_ndx, start_ndx+3):
    candidateInfo_tup = candidateInfo_list[nit_ndx]
    ct = getCt(candidateInfo_tup.series_uid)
    center_irc = xyz2irc(candidateInfo_tup.center_xyz, ct.origin_xyz, ct.vxSize_xyz, ct.direction_a)
    
    ct_list.append((ct, center_irc))
start_ndx = nit_ndx + 1

fig = plt.figure(figsize=(60,90))
subplot_ndx = 0 
for ct_ndx, (ct, center_irc) in enumerate(ct_list):
    mask_tup = build2dLungMask(ct.series_uid, int(center_irc.index))
    old_tup = old_build2dLungMask(ct.series_uid, int(center_irc.index))
    
#    ct_g = torch.from_numpy(ct.hu_a[int(center_irc.index)].astype(np.float32)).unsqueeze(0).unsqueeze(0).to('cuda')
#    pos_g = torch.from_numpy(ct.positive_mask[int(center_irc.index)].astype(np.float32)).unsqueeze(0).unsqueeze(0).to('cuda')
#    input_g = ct_g / 1000
    
#    label_g, neg_g, pos_g, lung_mask, mask_dict = mask_model(input_g, pos_g)
#    mask_tup = MaskTuple(**mask_dict)
    for attr_ndx, attr_str in enumerate(mask_tup._fields):

        subplot_ndx = 1 + 3 * 2 * attr_ndx + 2 * ct_ndx
        subplot = fig.add_subplot(len(mask_tup), len(ct_list)*2, subplot_ndx)
        subplot.set_title(attr_str)
        
        
        #print(layer_func, ct.hu_a.shape, layer_func(ct, mask_tup, int(center_irc.index)).shape, center_irc.index)

        plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 3000), cmap='RdGy')
        plt.imshow(mask_tup[attr_ndx][0][0].cpu(), clim=clim, cmap=tblue)

        subplot = fig.add_subplot(len(mask_tup), len(ct_list)*2, subplot_ndx+1)
        subplot.set_title('old '+ attr_str)

        plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 3000), cmap='RdGy')
        plt.imshow(old_tup[attr_ndx], clim=clim, cmap=tblue)
        
        #if attr_ndx == 1: break
    #break




# +
nit_ndx = 0
candidateInfo_tup = candidateInfo_list[nit_ndx]
ct = getCt(candidateInfo_tup.series_uid)
center_irc = xyz2irc(candidateInfo_tup.center_xyz, ct.origin_xyz, ct.vxSize_xyz, ct.direction_a)
print(candidateInfo_tup, 'center_irc', center_irc)

mask_tup = build2dLungMask(ct.series_uid, int(center_irc.index))
mask_tup = mask_tup._make(x.cpu().numpy()[0][0] for x in mask_tup)

print(mask_tup.pos_mask.sum() / (512*512))

plt.imshow(mask_tup.pos_mask)


# +
fig = plt.figure(figsize=(40,10))

#subplot = fig.add_subplot(1, 4, 1)
#subplot.set_title('ct', fontsize=30)
#plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 1000), cmap='gray')

subplot = fig.add_subplot(1, 4, 1)
subplot.set_title('ct', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')

#subplot = fig.add_subplot(1, 4, 3)
#subplot.set_title('ct annotation example', fontsize=30)
#plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 3000), cmap='gray')

#subplot = fig.add_subplot(1, 4, 2)
#subplot.set_title('raw_dense mask', fontsize=30)
#plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
#plt.imshow(mask_tup.dense_mask, clim=(0,1), cmap=tgray)

subplot = fig.add_subplot(1, 4, 2)
subplot.set_title('dense mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
plt.imshow(mask_tup.dense_mask, clim=(0,1), cmap=tgray)


#subplot = fig.add_subplot(1, 3, 2)
#subplot.set_title('denoise mask', fontsize=30)
#plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
#plt.imshow(mask_tup.denoise_mask, clim=(0,0.5), cmap=tgray)

subplot = fig.add_subplot(1, 4, 3)
subplot.set_title('body mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
plt.imshow(mask_tup.body_mask, clim=(0,1), cmap=tgray)

subplot = fig.add_subplot(1, 4, 4)
subplot.set_title('air mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
plt.imshow(mask_tup.air_mask, clim=(0,1), cmap=tgray)


# +
fig = plt.figure(figsize=(40,10))

subplot = fig.add_subplot(1, 4, 1)
subplot.set_title('lung mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
plt.imshow(mask_tup.lung_mask, clim=(0,1), cmap=tgray)

subplot = fig.add_subplot(1, 4, 2)
subplot.set_title('candidate mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
plt.imshow(mask_tup.candidate_mask, clim=(0,1), cmap=tgray)

subplot = fig.add_subplot(1, 4, 3)
subplot.set_title('ben mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
plt.imshow(mask_tup.neg_mask, clim=(0,1), cmap=tgray)

subplot = fig.add_subplot(1, 4, 4)
subplot.set_title('mal mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(ct.hu_a[int(center_irc.index)], clim=(-1000, 2000), cmap='gray')
plt.imshow(mask_tup.pos_mask, clim=(0,1), cmap=tgray)



# +
nit_ndx = 1
candidateInfo_tup = candidateInfo_list[nit_ndx]
ct = getCt(candidateInfo_tup.series_uid)
center_irc = xyz2irc(candidateInfo_tup.center_xyz, ct.origin_xyz, ct.vxSize_xyz, ct.direction_a)
print(candidateInfo_tup, 'center_irc', center_irc)

mask_tup = build2dLungMask(ct.series_uid, int(center_irc.index))
mask_tup = mask_tup._make(x.cpu().numpy()[0][0] for x in mask_tup)


fig = plt.figure(figsize=(20,20))

slice_a = ((ct.hu_a[int(center_irc.index)] / 1000) + 1) / 2
slice_a = slice_a.clip(0, 1)

subplot = fig.add_subplot(1, 1, 1)
subplot.set_title('mal mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)
plt.imshow(
    slice_a + 3 * slice_a * mask_tup.pos_mask, 
    #clim=(-2000, 2000), 
    cmap='gray',
)
#plt.imshow(ct.hu_a[int(center_irc.index)] * mask_tup.pos_mask, clim=(-1000,1000), cmap='gray')



# +
from p2ch13.training import LunaTrainingApp

cls_app = LunaTrainingApp(['--augmented', '--balanced'])
train_dl = cls_app.initTrainDl()

_ = train_dl.dataset[0]

# +
sample_ndx = 40

#while train_dl.dataset[sample_ndx][1][0]:
#    sample_ndx += 1

fig = plt.figure(figsize=(40,10))

candidate_t, pos_t, series_uid, center_irc = train_dl.dataset[sample_ndx]
subplot = fig.add_subplot(1, 4, 1)
subplot.set_title('augmented nodule 1', fontsize=30)
plt.imshow(candidate_t[0,12], clim=(-1000, 500), cmap='gray')

candidate_t, pos_t, series_uid, center_irc = train_dl.dataset[sample_ndx]
subplot = fig.add_subplot(1, 4, 2)
subplot.set_title('augmented nodule 2', fontsize=30)
plt.imshow(candidate_t[0,12], clim=(-1000, 500), cmap='gray')

candidate_t, pos_t, series_uid, center_irc = train_dl.dataset[sample_ndx]
subplot = fig.add_subplot(1, 4, 3)
subplot.set_title('augmented nodule 3', fontsize=30)
plt.imshow(candidate_t[0,12], clim=(-1000, 500), cmap='gray')

candidate_t, pos_t, series_uid, center_irc = train_dl.dataset[sample_ndx]
subplot = fig.add_subplot(1, 4, 4)
subplot.set_title('augmented nodule 4', fontsize=30)
plt.imshow(candidate_t[0,12], clim=(-1000, 500), cmap='gray')


print([sample_ndx, pos_t, series_uid, center_irc])


# +
print('a')
from p2ch13.dsets import TrainingLuna2dSegmentationDataset
print('yo')

ds = TrainingLuna2dSegmentationDataset(contextSlices_count=3, batch_size=4)
print('dawg')

input_t, label_int, raw_pos_t, series_uid, ct_ndx = ds[0]
print('sup')

plt.imshow(input_t[3])
print('yo')

# -


