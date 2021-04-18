import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from anomaly_detection.config import CURRENT_DIR, logger
from sklearn.neighbors import NearestNeighbors
from anomaly_detection.preprocessing.data import prepare_data
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

INPUT_DATASET_PATH = f"{CURRENT_DIR}/dataset/Train_data.csv"


def choose_optimum_params(data):
    neighbors = NearestNeighbors(n_neighbors=42)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.savefig(f"{CURRENT_DIR}/algorithms/plots/DensityBased.jpeg")
    plt.show()


def train_dbscan(data):
    EPSILON_DISTANCES = range(1, 20)

    models = []
    for epsilon_distance in EPSILON_DISTANCES:
        logger.info(f"DBSCAN clustering with epsilon distance = {epsilon_distance} started...")
        model = DBSCAN(eps=epsilon_distance).fit(data)
        logger.info(f"DBSCAN clustering with epsilon distance = {epsilon_distance} done...")

        models.append(model)

    predictions = []

    for model in models:
        logger.info(f"Prediction started...")
        prediction = model.labels_(data)
        logger.info(f"Prediction done...")
        predictions.append(prediction)

    fig, ax = plt.subplots()
    ax.plot(EPSILON_DISTANCES, predictions)
    plt.savefig(f"{CURRENT_DIR}/algorithms/plots/DensityBased.jpeg")
    plt.show()


if __name__ == "__main__":

    """
        0 - normal
        1 - anomaly
    """

    data = prepare_data(INPUT_DATASET_PATH)
    class_data = data["class"]
    data = data.loc[:, data.columns != "class"]
    data_copy = data.copy()

    model = DBSCAN(eps=10, min_samples=42).fit(data)

    NUM_OF_OUTLIERS = sum(1 for x in model.labels_ if x == -1)

    truth_values = [
        int(class_data[_idx]) for _idx, _ in enumerate(model.labels_) if model.labels_[_idx] == -1
    ]

    predicted_values = [
        1 for _ in range(NUM_OF_OUTLIERS)
    ]

    PRECISION_SCORE = precision_score(truth_values, predicted_values)
    RECALL_SCORE = recall_score(truth_values, predicted_values)
    F1_SCORE = f1_score(truth_values, predicted_values)
    ACCURACY_SCORE = accuracy_score(truth_values, predicted_values)

    print(f"PRECISION_SCORE={PRECISION_SCORE}")
    print(f"RECALL_SCORE={RECALL_SCORE}")
    print(f"F1_SCORE={F1_SCORE}")
    print(f"ACCURACY_SCORE={ACCURACY_SCORE}")
    print(f"NUM_OF_OUTLIERS={NUM_OF_OUTLIERS}")
