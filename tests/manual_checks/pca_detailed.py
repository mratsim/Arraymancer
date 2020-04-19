import numpy as np
from sklearn.decomposition import PCA

def test(X):
    pca = PCA(n_components=2)


    projected = pca.fit_transform(X)

    print(f'n_components: {pca.n_components_}')
    print(f'n_features: {pca.n_features_}')
    print(f'n_observations: {pca.n_samples_}')
    print(f'\nprojected:\n{projected}')
    print(f'\ncomponents:\n{pca.components_}')
    print(f'\nexplained_variance:\n{pca.explained_variance_}')
    print(f'\nexplained_variance_ratio:\n{pca.explained_variance_ratio_}')
    print(f'\nmean:\n{pca.mean_}')
    print(f'\nsingular_values:\n{pca.singular_values_}')
    print(f'\nnoise_variance: {pca.noise_variance_}')

X = np.array(
    [[-1, -1],
     [-2, -1],
     [-3, -2],
     [1, 1],
     [2, 1],
     [3, 2]])
test(X)

# n_components: 2
# n_features: 2
# n_observations: 6

# projected:
# [[ 1.38340578  0.2935787 ]
#  [ 2.22189802 -0.25133484]
#  [ 3.6053038   0.04224385]
#  [-1.38340578 -0.2935787 ]
#  [-2.22189802  0.25133484]
#  [-3.6053038  -0.04224385]]

# components:
# [[-0.83849224 -0.54491354]
#  [ 0.54491354 -0.83849224]]

# explained_variance:
# [7.93954312 0.06045688]

# explained_variance_ratio:
# [0.99244289 0.00755711]

# mean:
# [0. 0.]

# singular_values:
# [6.30061232 0.54980396]

# noise_variance: 0.0
print('--------------------------------------')

# X = np.array(
#     [[-1, -1, 1],
#      [-2, -1, 4],
#      [-3, -2, -10],
#      [1, 1, 0],
#      [2, 1, 7],
#      [3, 2, 3]])
# test(X)

# n_components: 2
# n_features: 3
# n_observations: 6

# projected:
# [[ 0.29798775  1.36325421]
#  [-2.25650538  3.13955756]
#  [11.41612855 -0.16713955]
#  [ 0.33140736 -1.58301663]
#  [-6.55502622 -0.06288375]
#  [-3.23399207 -2.68977185]]

# components:
# [[-0.27778488 -0.17755165 -0.94409267]
#  [-0.78737248 -0.52094113  0.32964362]]

# explained_variance:
# [37.80910168  4.29759759]

# explained_variance_ratio:
# [0.89665854 0.10191931]

# mean:
# [0.         0.         0.83333333]

# singular_values:
# [13.74938211  4.63551378]

# noise_variance: 0.059967392969596925
