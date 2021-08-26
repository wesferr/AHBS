import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

semantic = {
    "Weight": 0,  # peso
    "Height": 1,  # altura
    "WAIST_HT": 2,  # 'altura_cintura'
    "CROTCH_HT": 3,  # 'altura_virilha'
    "UP_ARM_GTH_L": 4,  # 'braço'
    "WAIST_GTH": 5,  # 'cintura'
    "ARM_LTH_L": 6,  # 'comp_braço'
    "THIGH_GTH_L_HZ": 7,  # 'coxa'
    "CR_SHOULDER": 8,  # 'ombro_a_ombro'
    "CALF_GTH_L": 9,  # 'panturrilha'
    "BUST_CHEST_GTH": 10,  # 'peito'
    "MID_NECK_GTH": 11,  # 'pescoço'
    "DIST_NECK_T_HIP": 12,  # 'pescoço_ao_quadril'
    "WRIST_GTH": 13,  # 'pulso'
    "HIP_GTH": 14,  # 'quadril'
    "Age": 15
}

MEASURE_NUM = 15


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
        self.data = np.full(MEASURE_NUM, np.nan)
        self.data[0] = weight
        self.data[1] = height
        self.missing_initials(self.data)

        ages = self.measures[:, 15].copy()

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
            
            return temporary_current.flatten()

        return self.current_measures

if __name__ == "__main__":
    pred = Predictor(age=19, weight=65, height=165)
    for i in range(100):
        print(pred.predict_next(denormalize=True))