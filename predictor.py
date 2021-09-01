import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

semantic = {
    "Weight": 0, # peso
    "Height": 1, # altura
    'INSEAM_L': 2,
    'ELBOW_GTH_L': 3,
    'HEAD_CIRC': 4,
    'WAIST_GTH': 5,
    "WAIST_T_BUTTOCK": 6,
    'KNEE_HT': 7,
    'NECK_AC_BACK_WTH_AL': 8,
    'ARM_LTH_L': 9,
    'THIGH_GTH_L_HZ': 10,
    'CR_SHOULDER_O_NECK': 11,
    'NECK_AT_BASE_GTH': 12,
    'DIST_NECK_T_HIP': 13,
    "Age": 14
}

MEASURE_NUM = 15


class Predictor():
    def __init__(self, measures):

        self.measures = np.load("./processed_data/life_selected.npy")

        data = np.load("./processed_data/measure.npz", allow_pickle=True)
        self.measure, self.mean_measure, self.std_measure, self.normalized_measure = data.values()
        self.imputer = IterativeImputer()
        self.imputer.fit(self.normalized_measure.T)

        self.initial_values = measures.copy()
        self.initial_age = self.current_age = measures[-1]
        self.missing_initials(measures)

        ages = self.measures[:, -1].copy()
        

        self.mean_measures = self.measures.mean()
        self.std_measures = self.measures.std()
        self.measures -= self.mean_measures
        self.measures /= self.std_measures

        self.curves = {}
        for element in semantic.keys():
            column = self.measures[:, semantic[element]]
            self.curves[element] = np.polynomial.Polynomial.fit(ages, column, deg=4)
            first = self.curves[element](self.initial_age)
            self.curves[element] = self.curves[element] - first

 
    def missing_initials(self, measures):
        
        age = measures[-1]
        measures = measures[:-1]
        
        measures[0] = (measures[0] ** (1.0/3.0) * 100)
        measures *= 10

        measures -= self.mean_measure.flatten()
        measures /= self.std_measure.flatten()
        measures = self.imputer.transform([measures])

        self.current_measures = measures

    def predict_next(self, delta_time=1, denormalize=False):

        previous_age = self.current_age
        self.current_age += delta_time
        

        for element in list(semantic.keys())[:-1]:
            delta_measure = self.curves[element](self.current_age) - self.curves[element](previous_age)
            self.current_measures[0][semantic[element]] += delta_measure * self.measures.std()

        if denormalize:
            temporary_current = self.current_measures.copy().transpose()
            temporary_current *= self.std_measure
            temporary_current += self.mean_measure

            temporary_current /= 10
            temporary_current[0] = ((temporary_current[0]/100)**3)
            
            return np.append(temporary_current.flatten(), self.current_age)

        return np.append(self.current_measures, self.current_age)

if __name__ == "__main__":
    data = np.full(MEASURE_NUM, np.nan)
    data[0] = 65
    data[1] = 165
    data[-1] = 19
    pred = Predictor(data)
    with np.printoptions(precision=3, suppress=True):
        for i in range(19,80):
            print(pred.predict_next(denormalize=True))