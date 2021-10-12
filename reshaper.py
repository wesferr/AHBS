import scipy
import time
import predictor
import numpy as np
import multiprocessing
from itertools import product, repeat


FACE_NUM = 25000
VERTICES_NUM = 12500
MEASURE_NUM = 15


def calcnorm(face, vertex):
    AB = np.array(vertex[face[1]]) - np.array(vertex[face[0]])
    AC = np.array(vertex[face[2]]) - np.array(vertex[face[0]])
    n = np.cross(AB, AC)
    return [(face[0], n), (face[1], n), (face[2], n)]

class Reshaper:

    def __init__(self, label="female"):
        data = np.load("./processed_data/{}_cp.npz".format(label), allow_pickle=True)
        self.cp, self.vertices, self.faces, self.normals = data.values()

        data = np.load("./processed_data/{}_measure.npz".format(label), allow_pickle=True)
        self.measure, self.mean_measure, self.std_measure, self.normalized_measure = data.values()

        data = np.load("./processed_data/{}_deformation.npz".format(label), allow_pickle=True)
        self.deformation_inverse, self.deformation, self.determinants = data.values()

        data = np.load("./processed_data/{}_mask.npz".format(label), allow_pickle=True)
        self.mask, self.rfe_mat = data.values()

        data = np.load("./processed_data/{}_deformation_basis.npz".format(label), allow_pickle=True)
        self.deformation_basis, self.deformation_coeff = data.values()

        self.vertices_deformation = np.load("./processed_data/{}_vertices_deformation.npy".format(label), allow_pickle=True)

        self.measure_to_deformation = np.load("./processed_data/{}_measure_to_deformation.npy".format(label), allow_pickle=True)


        vd = self.vertices_deformation.item()
        self.lu = scipy.sparse.linalg.splu(vd.transpose().dot(vd).tocsc())

    # local mapping using measure + rfe_mat
    def build_body(self, measures):
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

        normals = []
        vertexNormalLists = [[] for i in range(0, len(vertex))]

        for face in facet:
            AB = np.array(vertex[face[1]]) - np.array(vertex[face[0]])
            AC = np.array(vertex[face[2]]) - np.array(vertex[face[0]])
            n = np.cross(AB, AC)
            n /= np.linalg.norm(n)
            # adiciona a normal ao array de todas as normais de cada um dos vertics
            for i in range(0, 3):
                vertexNormalLists[face[i]].append(n)

        # soma os vetores normais das faces que o vertice faz parte, faz a media e normaliza
        for idx, normalList in enumerate(vertexNormalLists):
            normalSum = np.zeros(3)
            for normal in normalList:
                normalSum += normal
            normal = normalSum / float(len(normalList))
            normal /= np.linalg.norm(normal)
            normals.append(list(map(float, normal.tolist())))
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
