import numpy as np
import pandas as pd

from collections import defaultdict
from time import time

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from sklearn import manifold



def add_jitter_to_Y(Y, scale=1/10):
    sd1 = np.std(Y[:, 0]); sd2 = np.std(Y[:, 1])
    Y[:,0] = [np.random.normal(0, sd1*scale) + y1 for y1 in Y[:, 0]]
    Y[:,1] = [np.random.normal(0, sd2*scale) + y2 for y2 in Y[:, 1]]
    
    return Y


def reassign_colors(colors):
    colors_unique = list(set(colors))
    #reassigned color from 0 to --
    colors_rev = [colors_unique.index(c) for c in colors]
    #return colors2atc for legend
    colors2atc = {c: atcid2name[chr(colors_unique[c])] for c in colors_rev}
    
    return colors_rev, colors2atc


def generate_dummy_X(X):
    #sampling 1 by the frequence of 1 in X
    num_one = X.sum().sum(); num_total = X.shape[0] * X.shape[1]
    one_freq = num_one/num_total

    X_generated = defaultdict(list)
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            X_generated[X.columns[i]].append(np.random.choice([0, 1], p=[1-one_freq, one_freq]))
            
    return pd.DataFrame(X_generated)


def perform_tSNE_add_jitter(X, perplexity):
    t0 = time()    
    
    tsne = manifold.TSNE(
        n_components=2,
        init="random",
        random_state=0,
        perplexity=perplexity,
        n_iter=300,
    )
    Y = tsne.fit_transform(X)
    Y = add_jitter_to_Y(Y)
    
    t1 = time()
    print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
    
    return Y
    

def draw_ax_by_colors(Y, colors, ax):
    Y1 = Y[:, 0]; Y2 = Y[:, 1]
    
    for c in list(set(colors)):
        sample_indexes_c = [idx for idx, cs in enumerate(colors) if cs==c]
        Y1_c = [Y1[idx] for idx in sample_indexes_c]
        Y2_c = [Y2[idx] for idx in sample_indexes_c]
        ax.scatter(Y1_c, Y2_c, c=COLORS[c], s=SIZE, alpha=ALPHA, label=ATCNAMES[c])

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")



## main code ##
#import X, colors
datasources_used = ['BindingDB', 'PubChem', 'PDSP_Ki', 'ChEMBL', 'total']
#save X, colors as pickle files
import pickle
X_dict = {}; colors_dict = {}
for ds in datasources_used:
    with open(f'X_colors_for_tSNE/X_{ds}.p', 'rb') as f:
        X_dict[ds] = pickle.load(f)
    with open(f'X_colors_for_tSNE/colors_{ds}.p', 'rb') as f:
        colors_dict[ds] = pickle.load(f)
        
#tSNE analysis & PharmGKB parameters
perplexities = [5, 10, 30]
jitter_scale = 1/50

#global varaibles for plotting/legends
ALPHA = 0.7
SIZE = 30
atcid2name = {
    'A': 'ALIMENTARY TRACT AND METABOLISM',
    'B': 'BLOOD AND BLOOD FORMING ORGANS',
    'C': 'CARDIOVASCULAR SYSTEM',
    'D': 'DERMATOLOGICALS',
    'G': 'GENITO URINARY SYSTEM AND SEX HORMONES',
    'H': 'SYSTEMIC HORMONAL PREPARATIONS, EXCL. SEX HORMONES AND INSULINS',
    'J': 'ANTIINFECTIVES FOR SYSTEMIC USE',
    'L': 'ANTINEOPLASTIC AND IMMUNOMODULATING AGENTS',
    'M': 'MUSCULO-SKELETAL SYSTEM',
    'N': 'NERVOUS SYSTEM',
    'P': 'ANTIPARASITIC PRODUCTS, INSECTICIDES AND REPELLENTS',
    'R': 'RESPIRATORY SYSTEM',
    'S': 'SENSORY ORGANS',
    'V': 'VARIOUS'
}
COLORS = {
    0: 'darkgoldenrod',
    1: 'red',
    2: 'firebrick',
    3: 'tan',
    4: 'violet',
    5: 'thistle',
    6: 'teal',
    7: 'cyan',
    8: 'olive',
    9: 'yellow',
    10: 'darkmagenta',
    11: 'lime',
    12: 'royalblue',
    13: 'black'
}
ds_print = {
    'BindingDB': 'BindingDB',
    'PDSP_Ki': 'PDSP', 
    'ChEMBL': 'ChEMBL',
    'total': 'Total'
}
ATCNAMES = list(atcid2name.values())

#tSNE analysis and save plots
has_legend = False
for ds in datasources_used:
    #set X and colors for tSNE
    X = X_dict[ds]
    colors_alpha = colors_dict[ds]
        
    #set plt figure
    if has_legend:
        (fig, subplots) = plt.subplots(1,
                                    len(perplexities)+1,
                                    figsize=(7*(len(perplexities)+1), 4))
    else:
        (fig, subplots) = plt.subplots(1,
                                    len(perplexities)+1,
                                    figsize=(7*(len(perplexities)+1), 4))
    #set title for plot
    fig.suptitle(f'Datasource:{ds}')

    #get colors (0-13) and colors2atc
    colors, colors2atc = reassign_colors([ord(ca) for ca in colors_alpha])
    #generate dummy_X (randomly collected drug-target affinity data with the same collection frequency)
    X_generated = generate_dummy_X(X)
    
    #draw plots for each perplexity
    for i2, perplexity in enumerate(perplexities):
        ax = subplots[i2]
        Y = perform_tSNE_add_jitter(X, perplexity)   
        #draw ax by colors     
        ax.set_title(f"{ds_print[ds]}")  
        draw_ax_by_colors(Y, colors, ax)

    #add tSNE plot for genertaed X (perplexity=10)
    ax = subplots[len(perplexities)]
    Y_generated = perform_tSNE_add_jitter(X_generated, 10)
    #draw ax by colors
    ax.set_title("Randomly collected")
    draw_ax_by_colors(Y, colors, ax)

    #add legend
    if has_legend:
        ax.legend(loc = (1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(f'tSNE_{ds}_legend{has_legend}.png', dpi=300)