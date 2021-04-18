from sklearn.svm import OneClassSVM
import pandas as pd
from anomaly_detection.config import INPUT_DATASET_PATH
from anomaly_detection.preprocessing.data import prepare_data
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

if __name__ == "__main__":

    """
        0 - normal
        1 - anomaly
    """

    data = prepare_data(INPUT_DATASET_PATH)
    class_data = data["class"]
    data_new = data.loc[:, data.columns != "class"]
    data_copy = data.copy()

    OUTLIER_FRACTION = 0.1

    model = OneClassSVM(nu=0.95 * OUTLIER_FRACTION)
    model.fit(data_new)

    predictions = model.predict(data_new)

    data['anomaly'] = pd.Series(predictions)
    data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})
    data['anomaly'] = pd.Series(predictions)

    indexes = []
    predicted_values = []
    truth_values = []

    predicted_values = [
        1 for x in range(sum(1 for t in predictions if t == -1))
    ]

    for _idx, prediction in enumerate(predictions):
        if prediction == -1:
            indexes.append(_idx)

    truth_values = [
        int(not class_data[_idx]) for _idx in indexes
    ]

    PRECISION_SCORE = precision_score(truth_values, predicted_values)
    RECALL_SCORE = recall_score(truth_values, predicted_values)
    F1_SCORE = f1_score(truth_values, predicted_values)
    ACCURACY_SCORE = accuracy_score(truth_values, predicted_values)

    print(f"PRECISION_SCORE={PRECISION_SCORE}")
    print(f"RECALL_SCORE={RECALL_SCORE}")
    print(f"F1_SCORE={F1_SCORE}")
    print(f"ACCURACY_SCORE={ACCURACY_SCORE}")
    print(f"NUM_OF_OUTLIERS={len(predicted_values)}")
