import scipy
import time
import predictor
import numpy as np
import multiprocessing
from itertools import product, repeat


FACE_NUM = 25000
VERTICES_NUM = 12500
MEASURE_NUM = 14


def calcnorm(face, vertex):
    AB = np.array(vertex[face[1]]) - np.array(vertex[face[0]])
    AC = np.array(vertex[face[2]]) - np.array(vertex[face[0]])
    n = np.cross(AB, AC)
    return [(face[0], n), (face[1], n), (face[2], n)]

class Reshaper:

    def __init__(self):
        data = np.load("./processed_data/cp.npz", allow_pickle=True)
        self.cp, self.vertices, self.faces, self.normals = data.values()

        data = np.load("./processed_data/measure.npz", allow_pickle=True)
        self.measure, self.mean_measure, self.std_measure, self.normalized_measure = data.values()

        data = np.load("./processed_data/deformation.npz", allow_pickle=True)
        self.deformation_inverse, self.deformation, self.determinants = data.values()

        data = np.load("./processed_data/mask.npz", allow_pickle=True)
        self.mask, self.rfe_mat = data.values()

        data = np.load("./processed_data/deformation_basis.npz", allow_pickle=True)
        self.deformation_basis, self.deformation_coeff = data.values()

        self.vertices_deformation = np.load("./processed_data/vertices_deformation.npy", allow_pickle=True)

        self.measure_to_deformation = np.load("./processed_data/measure_to_deformation.npy", allow_pickle=True)


        vd = self.vertices_deformation.item()
        self.lu = scipy.sparse.linalg.splu(vd.transpose().dot(vd).tocsc())

    # local mapping using measure + rfe_mat
    def build_body(self, measures):
        if measures.size == 15:
            measures = measures[:-1]
        measures = measures.reshape(MEASURE_NUM, 1)
        measures *= self.std_measure
        measures += self.mean_measure

        d = []
        for i in range(0, FACE_NUM):
            mask = self.mask[:, i]
            alpha = measures[mask]
            d.extend(self.rfe_mat[i].dot(alpha))

        d = np.array(d)
        v, n, f = self.synthesize(d)
        return v, n, f

    def compute_normals(vertex, facet):

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()//2)
        tasks = [(
            facet[i],
            vertex
            ) for i in range(len(facet))]
        faces_normals = pool.starmap(calcnorm, tasks)
        

        elements = [[] for i in range(0, len(vertex))]
        for element in faces_normals:
            elements[element[0][0]].append(element[0][1])
            elements[element[1][0]].append(element[1][1])
            elements[element[2][0]].append(element[2][1])

        normals = []
        for normalList in elements:
            normal = sum(normalList) / len(normalList)
            normal /= np.linalg.norm(normal)
            normals.append(normal)

        

        return np.array(normals)

    # synthesize a body by deform-based, given deform, output vertex
    def synthesize(self, deform):
        Atd = self.vertices_deformation.item().transpose().dot(deform)
        vertices = self.lu.solve(Atd)
        vertices = vertices[:VERTICES_NUM * 3].reshape(VERTICES_NUM, 3)
        vertices -= np.mean(vertices, axis=0)
        normals = Reshaper.compute_normals(vertices, self.faces-1)
        return vertices, normals, self.faces-1

# save obj file
def save_obj(filename, v, f, n):
    file = open(filename, 'w')
    for i in range(0, v.shape[0]):
        file.write('v %f %f %f\n' % (v[i][0], v[i][1], v[i][2]))
    for i in range(0, v.shape[0]):
        file.write('vn %f %f %f\n' % (n[i][0], n[i][1], n[i][2]))
    for i in range(0, f.shape[0]):
        file.write('f %d//%d %d//%d %d//%d\n' % (f[i][0]+1, f[i][0]+1, f[i][1]+1, f[i][1]+1, f[i][2]+1, f[i][2]+1))
    file.close()
    print('obj file {} saved'.format(filename))


if __name__ == "__main__":

    pred = predictor.Predictor(age=19, weight=85, height=180)
    reshaper = Reshaper()
    v, n, f = reshaper.build_body(pred.current_measures)
    save_obj("teste.obj", v, f, n)
