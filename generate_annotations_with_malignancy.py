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

# # Malignancy Annotations
#
# This notebook compiles the `annotations_with_malignancy.csv` and also drops annotations for CTs it cannot find.
#
# In addition to the usual suspects, you need to have the `pylidc` Python package (use `pip install pylidc` or [check out the source](https://pylidc.github.io/).

import torch
import SimpleITK as sitk
import pandas
import glob, os
import numpy
import tqdm
import pylidc


# We first load the annotations from the LUNA challenge.

annotations = pandas.read_csv('data/part2/luna/annotations.csv')

# For the CTs where we have a `.mhd` file, we collect the malignancy_data from PyLIDC.
#
# It is a bit tedious as we need to convert the pixel locations provided by PyLIDC to physical points.
# We will see some warnings about annotations to be too close too each other (PyLIDC expects to have 4 annotations per site, see Chapter 14 for some details, including when we consider a nodule to be malignant).
#
# This takes quite a while (~1-2 seconds per scan on one of the author's computer).

malignancy_data = []
missing = []
spacing_dict = {}
scans = {s.series_instance_uid:s for s in pylidc.query(pylidc.Scan).all()}
suids = annotations.seriesuid.unique()
for suid in tqdm.tqdm(suids):
    fn = glob.glob('./data-unversioned/part2/luna/subset*/{}.mhd'.format(suid))
    if len(fn) == 0 or '*' in fn[0]:
        missing.append(suid)
        continue
    fn = fn[0]
    x = sitk.ReadImage(fn)
    spacing_dict[suid] = x.GetSpacing()
    s = scans[suid]
    for ann_cluster in s.cluster_annotations():
        # this is our malignancy criteron described in Chapter 14
        is_malignant = len([a.malignancy for a in ann_cluster if a.malignancy >= 4])>=2
        centroid = numpy.mean([a.centroid for a in ann_cluster], 0)
        bbox = numpy.mean([a.bbox_matrix() for a in ann_cluster], 0).T
        coord = x.TransformIndexToPhysicalPoint([int(numpy.round(i)) for i in centroid[[1, 0, 2]]])
        bbox_low = x.TransformIndexToPhysicalPoint([int(numpy.round(i)) for i in bbox[0, [1, 0, 2]]])
        bbox_high = x.TransformIndexToPhysicalPoint([int(numpy.round(i)) for i in bbox[1, [1, 0, 2]]])
        malignancy_data.append((suid, coord[0], coord[1], coord[2], bbox_low[0], bbox_low[1], bbox_low[2], bbox_high[0], bbox_high[1], bbox_high[2], is_malignant, [a.malignancy for a in ann_cluster]))


# You can check how many `mhd`s you are missing. It seems that the LUNA data has dropped a couple(?). Don't worry if there are <10 missing.

print("MISSING", missing)

# We stick the data we got from PyLIDC into a DataFrame.

df_mal = pandas.DataFrame(malignancy_data, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'bboxLowX', 'bboxLowY', 'bboxLowZ', 'bboxHighX', 'bboxHighY', 'bboxHighZ', 'mal_bool', 'mal_details'])

# And now we match the malignancy data to the annotations. This is a lot faster...

processed_annot = []
annotations['mal_bool'] = float('nan')
annotations['mal_details'] = [[] for _ in annotations.iterrows()]
bbox_keys = ['bboxLowX', 'bboxLowY', 'bboxLowZ', 'bboxHighX', 'bboxHighY', 'bboxHighZ']
for k in bbox_keys:
    annotations[k] = float('nan')
for series_id in tqdm.tqdm(annotations.seriesuid.unique()):
    # series_id = '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860'
    # c = candidates[candidates.seriesuid == series_id]
    a = annotations[annotations.seriesuid == series_id]
    m = df_mal[df_mal.seriesuid == series_id]
    if len(m) > 0:
        m_ctrs = m[['coordX', 'coordY', 'coordZ']].values
        a_ctrs = a[['coordX', 'coordY', 'coordZ']].values
        #print(m_ctrs.shape, a_ctrs.shape)
        matches = (numpy.linalg.norm(a_ctrs[:, None] - m_ctrs[None], ord=2, axis=-1) / a.diameter_mm.values[:, None] < 0.5)
        has_match = matches.max(-1)
        match_idx = matches.argmax(-1)[has_match]
        a_matched = a[has_match].copy()
        # c_matched['diameter_mm'] = a.diameter_mm.values[match_idx]
        a_matched['mal_bool'] = m.mal_bool.values[match_idx]
        a_matched['mal_details'] = m.mal_details.values[match_idx]
        for k in bbox_keys:
            a_matched[k] = m[k].values[match_idx]
        processed_annot.append(a_matched)
        processed_annot.append(a[~has_match])
    else:
        processed_annot.append(c)
processed_annot = pandas.concat(processed_annot)
processed_annot.sort_values('mal_bool', ascending=False, inplace=True)
processed_annot['len_mal_details'] = processed_annot.mal_details.apply(len)

# Finally, we drop NAs (where we didn't find a match) and save it in the right place.

df_nona = processed_annot.dropna()
df_nona.to_csv('./data/part2/luna/annotations_with_malignancy.csv', index=False)


