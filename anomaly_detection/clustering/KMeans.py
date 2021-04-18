from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from anomaly_detection.config import CURRENT_DIR, logger, INPUT_DATASET_PATH
from anomaly_detection.clustering.utils import calculate_distance
from anomaly_detection.preprocessing.data import prepare_data
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def train_kmeans(data):
    NUMBER_OF_CLUSTERS = range(1, 40)

    models = []
    for num_cluster in NUMBER_OF_CLUSTERS:
        logger.info(f"K-means clustering with number of cluster = {num_cluster} started...")
        model = KMeans(n_clusters=num_cluster).fit(data)
        logger.info(f"K-means clustering with number of cluster = {num_cluster} done...")

        models.append(model)

    predictions = []

    for model in models:
        logger.info(f"Prediction started...")
        prediction = model.score(data)
        logger.info(f"Prediction done...")
        predictions.append(prediction)

    fig, ax = plt.subplots()
    ax.plot(NUMBER_OF_CLUSTERS, predictions)
    plt.savefig(f"{CURRENT_DIR}/algorithms/plots/Kmeans-scores.jpeg")
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

    _best_model = KMeans(n_clusters=3).fit(data)
    data['cluster'] = _best_model.predict(data)

    distances = calculate_distance(data_copy, _best_model)

    OUTLIER_FRACTION = 0.01
    NUM_OF_OUTLIERS = int(OUTLIER_FRACTION * len(distances))
    OUTLIERS = distances[-NUM_OF_OUTLIERS:]

    truth_values = [
        int(not class_data[_idx]) for (_idx, _) in OUTLIERS
    ]

    predicted_values = [
        1 for _ in OUTLIERS
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
