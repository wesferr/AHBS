import numpy as np
from matplotlib import pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

semantic = {
    "Weight": 0,
    "Height": 1,
    "MID_NECK_GTH": 2,
    "BUST_CHEST_GTH": 3,
    "BELLY_CIRC": 4,
    "MIDDLE_HIP": 5,
    "ARM_LTH_T_NECK_L": 6,
    "CROTCH_HT": 7,
    "AC_BACK_WTH_AL": 8,
    "DIST_NECK_T_HIP": 9,
    "WAIST_GTH": 10,
    "HIP_GTH": 11,
    "WAIST_HT": 12,
    "ARM_LTH_L": 13,
    "UP_ARM_GTH_L": 14,
    "WRIST_GTH": 15,
    "BREAST_HT": 16,
    "KNEE_GTH_L": 17,
    "WTH_THIGH_SUM": 18,
}


class Predictor():
    def __init__(self, age, weight, height):

        self.measures = np.load("./processed_data/life_selected.npy")

        data = np.load("./processed_data/measure.npz", allow_pickle=True)
        self.measure, self.mean_measure, self.std_measure, self.normalized_measure = data.values()
        self.imputer = IterativeImputer()
        self.imputer.fit(self.normalized_measure.T)

        self.initial_age = age
        self.initial_weight = weight
        self.initial_height = height
        self.current_age = age
        data = np.full(19, np.nan)
        data[0] = weight
        data[1] = height
        self.missing_initials(data)

        ages = self.measures[:, 19].copy()
        age_array = np.arange(19,85)

        self.mean_measures = self.measures.mean()
        self.std_measures = self.measures.std()
        self.measures -= self.mean_measures
        self.measures /= self.std_measures

        self.curves = {}
        for element in semantic.keys():
            self.curves[element] = np.polynomial.Polynomial.fit(ages, self.measures[:, semantic[element]], deg=4)
            first = self.curves[element](self.initial_age)
            self.curves[element] = self.curves[element] - first
 
    def missing_initials(self, measures):
        
        measures[0] = (measures[0] ** (1.0/3.0) * 100)
        measures *= 10

        measures -= self.mean_measure.flatten()
        measures /= self.std_measure.flatten()
        measures = self.imputer.transform([measures])

        self.current_measures = measures

    def predict_next(self, delta_time=1, denormalize=False):

        previous_age = self.current_age
        self.current_age += delta_time

        for element in semantic.keys():
            delta_measure = self.curves[element](self.current_age) - self.curves[element](previous_age)
            self.current_measures[0][semantic[element]] += delta_measure * self.measures.std()

        if denormalize:
            temporary_current = self.current_measures.copy().transpose()
            temporary_current *= self.std_measure
            temporary_current += self.mean_measure

            temporary_current /= 10
            temporary_current[0] = ((temporary_current[0]/100)**3)
            
            return temporary_current.flatten()

        return self.current_measures

if __name__ == "__main__":
    pred = Predictor(age=19, weight=65, height=175)
    for i in range(10):
        print(pred.predict_next(1))