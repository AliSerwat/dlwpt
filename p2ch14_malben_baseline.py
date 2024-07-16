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

# +
import torch
# %matplotlib inline
from matplotlib import pyplot

import p2ch14.dsets
import p2ch14.model

# -

ds = p2ch14.dsets.MalignantLunaDataset(val_stride=10, isValSet_bool=True)  # <1>
nodules = ds.ben_list + ds.mal_list
is_mal = torch.tensor([n.isMal_bool for n in nodules])  # <2>
diam  = torch.tensor([n.diameter_mm for n in nodules])
num_mal = is_mal.sum()  # <3>
num_ben = len(is_mal) - num_mal

threshold = torch.linspace(diam.max(), diam.min())

predictions = (diam[None] >= threshold[:, None])  # <1>
tp_diam = (predictions & is_mal[None]).sum(1).float() / num_mal  # <2>
fp_diam = (predictions & ~is_mal[None]).sum(1).float() / num_ben

fp_diam_diff =  fp_diam[1:] - fp_diam[:-1]
tp_diam_avg  = (tp_diam[1:] + tp_diam[:-1])/2
auc_diam = (fp_diam_diff * tp_diam_avg).sum()

# +
fp_fill = torch.ones((fp_diam.shape[0] + 1,))
fp_fill[:-1] = fp_diam

tp_fill = torch.zeros((tp_diam.shape[0] + 1,))
tp_fill[:-1] = tp_diam

print(threshold)
print(fp_diam)
print(tp_diam)
# -

for i in range(threshold.shape[0]):
    print(i, threshold[i], fp_diam[i], tp_diam[i])

pyplot.figure(figsize=(7,5), dpi=1200)
for i in [62, 88]:
    pyplot.scatter(fp_diam[i], tp_diam[i], color='red')
    print(f'diam: {round(threshold[i].item(), 2)}, x: {round(fp_diam[i].item(), 2)}, y: {round(tp_diam[i].item(), 2)}')
pyplot.fill(fp_fill, tp_fill, facecolor='#0077bb', alpha=0.25)
pyplot.plot(fp_diam, tp_diam, label=f'diameter baseline, AUC={auc_diam:.3f}')
pyplot.title(f'ROC diameter baseline, AUC={auc_diam:.3f}')
pyplot.ylabel('true positive rate')
pyplot.xlabel('false positive rate')
pyplot.savefig('roc_diameter_baseline.png')

model = p2ch14.model.LunaModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sd = torch.load('data/part2/models/cls_2020-02-08_01.19.40_finetune-head.best.state')
model.load_state_dict(sd['model_state'])
model.to(device)
model.eval();

ds = p2ch14.dsets.MalignantLunaDataset(val_stride=10, isValSet_bool=True)
dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=4)


preds = []
truth = []
for inp, label, _, _, _ in dl:
    inp = inp.to(device)    
    truth += (label[:,1]>0).tolist()
    with torch.no_grad():
        _, p = model(inp)
        preds += p[:, 1].tolist()
truth = torch.tensor(truth)
preds = torch.tensor(preds)

# +
num_mal = truth.sum()
num_ben = len(truth) - num_mal
threshold = torch.linspace(1, 0)
tp_finetune = ((preds[None] >= threshold[:, None]) & truth[None]).sum(1).float() / num_mal
fp_finetune = ((preds[None] >= threshold[:, None]) & ~truth[None]).sum(1).float() / num_ben
fp_finetune_diff = fp_finetune[1:]-fp_finetune[:-1]
tp_finetune_avg  = (tp_finetune[1:]+tp_finetune[:-1])/2
auc_finetune = (fp_finetune_diff * tp_finetune_avg).sum()

pyplot.figure(figsize=(7,5), dpi=300)
pyplot.fill(fp_fill, tp_fill, facecolor='#0077bb', alpha=0.25)
pyplot.plot(fp_diam, tp_diam, label=f'diameter baseline, AUC={auc_diam:.3f}')
pyplot.plot(fp_finetune, tp_finetune, label=f'1 layer fine-tuned, AUC={auc_finetune:.3f}')
pyplot.legend()
pyplot.savefig('roc_finetune.png')
# -

if 1:
    fn = 'data/part2/models/cls_2020-02-08_00.19.45_finetune-depth2.best.state'
    model = p2ch14.model.LunaModel()
    sd = torch.load(fn, map_location='cpu')['model_state']
    model.load_state_dict(sd)
    model.to(device)
    model.eval();


model.eval()
preds = []
truth = []
for inp, label, _, _, _ in dl:
    inp = inp.to(device)    
    truth += (label[:,1]>0).tolist()
    with torch.no_grad():
        _, p = model(inp)
        preds += p[:, 1].tolist()
truth = torch.tensor(truth)
preds = torch.tensor(preds)

# +
num_mal = truth.sum()
num_ben = len(truth) - num_mal
threshold = torch.linspace(1, 0)
tp = ((preds[None] >= threshold[:, None]) & truth[None]).sum(1).float() / num_mal
fp = ((preds[None] >= threshold[:, None]) & ~truth[None]).sum(1).float() / num_ben

fp_diff = fp[1:]-fp[:-1]
tp_avg  = (tp[1:]+tp[:-1])/2
auc_modified = (fp_diff * tp_avg).sum()

pyplot.figure(figsize=(7,5), dpi=300)
pyplot.fill(fp_fill, tp_fill, facecolor='#0077bb', alpha=0.25)
pyplot.plot(fp_diam, tp_diam, label=f'diameter baseline, AUC={auc_diam:.3f}')
pyplot.plot(fp_finetune, tp_finetune, label=f'1 layer fine-tuned, AUC={auc_finetune:.3f}')
pyplot.plot(fp, tp, label=f'2 layers fine-tuned, AUC={auc_modified:.3f}')
pyplot.legend()
pyplot.savefig('roc_modified.png')
# -




