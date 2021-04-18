import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def prepare_data(input_file: str):

    data = pd.read_csv(input_file)
    ordinal_encoder = OrdinalEncoder()

    STRING_COLUMNS = [
        "protocol_type",
        "service",
        "flag",
        "class"
    ]

    for column in STRING_COLUMNS:

        data[column] = ordinal_encoder.fit_transform(
            data[column].to_numpy().reshape(-1, 1)
        ).astype("int64")

    def _normalize(df):
        result = df.copy()

        NORMALIZED_COLUMNS = [
            "src_bytes",
            "dst_bytes",
            "count",
            "srv_count",
            "dst_host_count",
            "dst_host_srv_count"
        ]

        for feature_name in NORMALIZED_COLUMNS:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()

            if isinstance(max_value, str):
                continue

            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

        df["protocol_type"] = df.protocol_type.astype("category")
        return result

    return _normalize(data)
