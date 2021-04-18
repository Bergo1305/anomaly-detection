import math
from itertools import repeat
from matplotlib import pyplot as plt
from anomaly_detection.preprocessing.data import prepare_data
from anomaly_detection.config import INPUT_DATASET_PATH, logger, CURRENT_DIR
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


class HBOS(object):
    def __init__(
            self, log_scale=True, ranked=False, bin_info_array=None, mode_array=None, nominal_array=None
    ):
        if bin_info_array is None:
            self.bin_info_array = []

        if mode_array is None:
            self.mode_array = []

        if nominal_array is None:
            self.nominal_array = []

        self.log_scale = log_scale
        self.ranked = ranked
        self.histogram_list = []

    def fit(self, data):
        attr_size = len(data.columns)
        total_data_size = len(data)

        if len(self.bin_info_array) == 0:
            self.bin_info_array = list(repeat(-1, attr_size))

        if len(self.mode_array) == 0:
            self.mode_array = list(repeat('dynamic binwidth', attr_size))

        if len(self.nominal_array) == 0:
            self.nominal_array = list(repeat(False, attr_size))

        if self.ranked:
            self.log_scale = False

        normal = 1.0

        for i in range(len(self.bin_info_array)):
            if self.bin_info_array[i] == -1:
                self.bin_info_array[i] = round(math.sqrt(len(data)))

        self.histogram_list = []
        for i in range(attr_size):
            self.histogram_list.append([])

        maximum_value_of_rows = data.apply(max).values

        sorted_data = data.apply(sorted)

        for attrIndex in range(len(sorted_data.columns)):
            attr = sorted_data.columns[attrIndex]
            last = 0
            bin_start = sorted_data[attr][0]
            if self.mode_array[attrIndex] == 'dynamic binwidth':
                if self.nominal_array[attrIndex]:
                    while last < len(sorted_data) - 1:
                        last = self.create_dynamic_histogram(self.histogram_list, sorted_data, last, 1, attrIndex, True)
                else:
                    length = len(sorted_data)
                    binwidth = self.bin_info_array[attrIndex]
                    while last < len(sorted_data) - 1:
                        values_per_bin = math.floor(len(sorted_data) / self.bin_info_array[attrIndex])
                        last = self.create_dynamic_histogram(self.histogram_list, sorted_data, last, values_per_bin,
                                                             attrIndex, False)
                        if binwidth > 1:
                            length = length - self.histogram_list[attrIndex][-1].quantity
                            binwidth = binwidth - 1
            else:
                count_bins = 0
                binwidth = (sorted_data[attr][len(sorted_data) - 1] - sorted_data[attr][0]) * 1.0 / self.bin_info_array[
                    attrIndex]
                if (self.nominal_array[attrIndex]) | (binwidth == 0):
                    binwidth = 1
                while last < len(sorted_data):
                    is_last_bin = count_bins == self.bin_info_array[attrIndex] - 1
                    last = self.create_static_histogram(self.histogram_list, sorted_data, last, binwidth, attrIndex,
                                                        bin_start, is_last_bin)
                    bin_start = bin_start + binwidth
                    count_bins = count_bins + 1

        max_score = []

        for i in range(len(self.histogram_list)):
            max_score.append(0)
            histogram = self.histogram_list[i]

            for k in range(len(histogram)):
                _bin = histogram[k]
                _bin.total_data_size = total_data_size
                _bin.calc_score(maximum_value_of_rows[i])
                if max_score[i] < _bin.score:
                    max_score[i] = _bin.score

        for i in range(len(self.histogram_list)):
            histogram = self.histogram_list[i]
            for k in range(len(histogram)):
                _bin = histogram[k]
                _bin.normalize_score(normal, max_score[i], self.log_scale)

    def predict(self, data):
        score_array = []
        for i in range(len(data)):
            each_data = data.values[i]
            value = 1
            if self.log_scale | self.ranked:
                value = 0
            for attr in range(len(data.columns)):
                score = self.get_score(self.histogram_list[attr], each_data[attr])
                if self.log_scale:
                    value = value + score
                elif self.ranked:
                    value = value + score
                else:
                    value = value * score
            score_array.append(value)
        return score_array

    def fit_predict(self, data):
        self.fit(data)
        return self.predict(data)

    @staticmethod
    def get_score(histogram, value):
        for i in range(len(histogram) - 1):
            _bin = histogram[i]
            if (_bin.range_from <= value) & (value < _bin.range_to):
                return _bin.score

        _bin = histogram[-1]
        if (_bin.range_from <= value) & (value <= _bin.range_to):
            return _bin.score
        return 0

    @staticmethod
    def check_amount(sorted_data, first_occurrence, values_per_bin, attr):
        if first_occurrence + values_per_bin < len(sorted_data):
            if sorted_data[attr][first_occurrence] == sorted_data[attr][first_occurrence + values_per_bin]:
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def create_dynamic_histogram(histogram_list, sorted_data, first_index, values_per_bin, attr_index, is_nominal):
        attr = sorted_data.columns[attr_index]

        _bin = HistogramBin(sorted_data[attr][first_index], 0, 0)

        if first_index + values_per_bin < len(sorted_data):
            last_index = first_index + values_per_bin
        else:
            last_index = len(sorted_data)

        _bin.add_quantitiy(1)

        cursor = first_index
        for i in range(int(first_index + 1), int(last_index)):
            if sorted_data[attr][i] == sorted_data[attr][cursor]:
                _bin.add_quantitiy(1)
                cursor = cursor + 1
            else:
                if HBOS.check_amount(sorted_data, i, values_per_bin, attr):
                    break
                else:
                    _bin.add_quantitiy(1)
                    cursor = cursor + 1

        # continue to put values in the _bin until a new values arrive
        for i in range(cursor + 1, len(sorted_data)):
            if sorted_data[attr][i] == sorted_data[attr][cursor]:
                _bin.quantity = _bin.quantity + 1
                cursor = cursor + 1
            else:
                break

        if cursor + 1 < len(sorted_data):
            _bin.range_to = sorted_data[attr][cursor + 1]
        else:
            if is_nominal:
                _bin.range_to = sorted_data[attr][len(sorted_data) - 1] + 1
            else:
                _bin.range_to = sorted_data[attr][len(sorted_data) - 1]

        if _bin.range_to - _bin.range_from > 0:
            histogram_list[attr_index].append(_bin)
        elif len(histogram_list[attr_index]) == 0:
            _bin.range_to = _bin.range_to + 1
            histogram_list[attr_index].append(_bin)
        else:
            last_bin = histogram_list[attr_index][-1]
            last_bin.add_quantitiy(_bin.quantity)
            last_bin.range_to = _bin.range_to

        return cursor + 1

    @staticmethod
    def create_static_histogram(histogram_list, sorted_data, first_index, binwidth, attr_index, bin_start, last_bin):
        attr = sorted_data.columns[attr_index]
        _bin = HistogramBin(bin_start, bin_start + binwidth, 0)
        if last_bin:
            _bin = HistogramBin(bin_start, sorted_data[attr][len(sorted_data) - 1], 0)

        last = first_index - 1
        cursor = first_index

        while True:
            if cursor >= len(sorted_data):
                break
            if sorted_data[attr][cursor] > _bin.range_to:
                break
            _bin.quantity = _bin.quantity + 1
            last = cursor
            cursor = cursor + 1

        histogram_list[attr_index].append(_bin)
        return last + 1


class HistogramBin:
    def __init__(self, range_from, range_to, quantity):
        self.range_from = range_from
        self.range_to = range_to
        self.quantity = quantity
        self.score = 0
        self.total_data_size = 0

    def get_height(self):
        width = self.range_to - self.range_from
        height = self.quantity / width
        return height

    def add_quantitiy(self, anz):
        self.quantity = self.quantity + anz

    def calc_score(self, max_score):
        if max_score == 0:
            max_score = 1

        if self.quantity > 0:
            self.score = 1.0 * self.quantity / (
                (self.range_to - self.range_from) * self.total_data_size * 1.0 / abs(max_score))

    def normalize_score(self, normal, max_score, log_scale):
        self.score = self.score * normal / max_score
        if self.score == 0:
            return
        self.score = 1 / self.score
        if log_scale:
            self.score = math.log10(self.score)


if __name__ == "__main__":

    data = prepare_data(INPUT_DATASET_PATH)
    class_data = data["class"]
    data_new = data.loc[:, data.columns != "class"]

    hbos = HBOS()
    result = hbos.fit_predict(data_new)

    data['hbos'] = result
    hbos_top1000_data = data.sort_values(by=['hbos'], ascending=False)[:1000]

    _indexes = [
        index for index in hbos_top1000_data.index
    ]

    truth_values = [
        int(not class_data[_idx]) for _idx in _indexes
    ]

    predicted_values = [
        1 for _ in range(len(_indexes))
    ]

    PRECISION_SCORE = precision_score(truth_values, predicted_values)
    RECALL_SCORE = recall_score(truth_values, predicted_values)
    F1_SCORE = f1_score(truth_values, predicted_values)
    ACCURACY_SCORE = accuracy_score(truth_values, predicted_values)

    print(f"PRECISION_SCORE={PRECISION_SCORE}")
    print(f"RECALL_SCORE={RECALL_SCORE}")
    print(f"F1_SCORE={F1_SCORE}")
    print(f"ACCURACY_SCORE={ACCURACY_SCORE}")
    print(f"NUM_OF_OUTLIERS={1000}")

    plt.scatter(range(1000), hbos_top1000_data['class'].cumsum(), marker='1')
    plt.xlabel('Top N data')
    plt.ylabel('Anomalies found')
    plt.savefig(f"{CURRENT_DIR}/statistical/hbos.jpeg")
    plt.show()




