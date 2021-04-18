import numpy as np


def calculate_distance(data, model):
    distances = []
    for i in range(len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i] - 1]

        distances.append((i, np.linalg.norm(Xa-Xb)))

    distances = sorted(distances, key=lambda _tuple: _tuple[1])
    return distances

