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
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# Make precision and recall data.
range_a = np.arange(0.01, 1, 0.01)
precision_a, recall_a = np.meshgrid(range_a, range_a)

f1_score = np.sqrt(2 * precision_a * recall_a / (precision_a + recall_a))

def plotScore(title_str, other_score):
    fig, subplts = plt.subplots(nrows=1, ncols=1, dpi=300, figsize=(7/2, 2.5))

    subplts.set_title(title_str + "(p, r)")
    subplts.contourf(other_score, cmap='gray')

    subplts.set_xlabel("precision")
    subplts.set_ylabel("recall")

    fig.tight_layout()

    plt.show()
    
def plotScores(title_str, other_score):
    fig, subplts = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(7, 3.5))

    subplts[0].set_title(title_str + "(p, r)")
    subplts[0].contourf(other_score, cmap='gray')

    subplts[1].set_title("f1(p, r)")
    subplts[1].contourf(f1_score, cmap='gray')

    #subplts[2].set_title("f1 - " + title_str)
    #subplts[2].contourf(f1_score - other_score, cmap='gray')

    for subplt in subplts:
        subplt.set_xlabel("precision")
        subplt.set_ylabel("recall")

    fig.tight_layout()

    plt.show()


# -

add_score = (precision_a + recall_a) / 2
plotScores("avg", add_score)

min_score = np.min(np.array([precision_a, recall_a]), axis=0)
plotScores("min", min_score)

mult_score = precision_a * recall_a
plotScores("mult", mult_score)

sqrt_score = np.sqrt(precision_a * recall_a)
plotScores("sqrt", sqrt_score)
