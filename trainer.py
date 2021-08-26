import numpy as np
import time
import sys
import scipy
from loader import Loader
import os
from multiprocessing import Pool, cpu_count
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

FACE_NUM = 25000
VERTICES_NUM = 12500
MEASURE_NUM = 15
D_BASIS_NUM = 10
V_BASIS_NUM = 10


class Trainer:

    # import the 4th point of the triangle, and calculate the deformation
    def assemble_face(v1, v2, v3):
        v21 = v2 - v1
        v31 = v3 - v1
        v41 = np.cross(v21, v31)
        v41 /= np.sqrt(np.linalg.norm(v41))
        return np.column_stack((v21, np.column_stack((v31, v41))))

    def get_inv_mean(mean_vertex, faces):
        print("Generating mean body deformation matrix")
        d_inv_mean = np.zeros((FACE_NUM, 3, 3))
        for i in range(FACE_NUM):
            f = faces[i] - 1
            v1 = mean_vertex[f[0]]
            v2 = mean_vertex[f[1]]
            v3 = mean_vertex[f[2]]
            d_inv_mean[i] = Trainer.assemble_face(v1, v2, v3)
            d_inv_mean[i] = np.linalg.inv(d_inv_mean[i])
        return d_inv_mean

    def generate_face_deformation(vertices, faces):
        print("Generating face deformation matrix")
        mean_vertex = vertices.mean(axis=0)
        deformation_inverse = Trainer.get_inv_mean(mean_vertex, faces)
        deformation = np.zeros((vertices.shape[0], FACE_NUM, 9))
        determinants = []

        initial_time = time.time()
        for i in range(FACE_NUM):

            f = faces[i] - 1
            for j in range(vertices.shape[0]):
                v1 = vertices[j, f[0]]
                v2 = vertices[j, f[1]]
                v3 = vertices[j, f[2]]
                Q = Trainer.assemble_face(v1, v2, v3).dot(deformation_inverse[i])
                determinants.append(np.linalg.det(Q))
                deformation[j, i, :] = Q.flat

            sys.stdout.write("{:.2f}% - {:.2f}s \r".format((i/FACE_NUM)*100, time.time() - initial_time))
            sys.stdout.flush()

        determinants = np.array(determinants).reshape(FACE_NUM, vertices.shape[0])

        return deformation_inverse, deformation, determinants

    def rfe_local(determinants, deformation, measures, normalized_measures, label="female", k_features=9):
        print("Running recursive feature elimination")
        body_num = deformation.shape[0]
        x = normalized_measures.transpose()
        initial_time = time.time()

        pool = Pool(processes=cpu_count())
        tasks = [(
            i,
            determinants[i, :],
            deformation[:, i, :],
            body_num,
            x,
            measures,
            k_features, initial_time) for i in range(FACE_NUM)]

        results = pool.starmap(Trainer.rfe_multiprocess, tasks)

        pool.close()
        pool.join()

        rfe_mat = np.array([ele[0] for ele in results]).reshape(FACE_NUM, 9, k_features)
        mask = np.array([ele[1] for ele in results]).reshape(FACE_NUM, MEASURE_NUM).transpose()
        return mask, rfe_mat

    def rfe_multiprocess(i, determinants, deformation, body_num, x, measure, k_features, initial_time):
        y = np.array(determinants)
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=k_features)
        rfe.fit(x, y)
        flag = np.array(rfe.support_).reshape(MEASURE_NUM, 1)
        flag = flag.repeat(body_num, axis=1)

        # calculte linear mapping mat
        S = np.array(deformation)
        S.shape = (S.size, 1)
        m = np.array(measure[flag])
        m.shape = (k_features, body_num)

        M = Trainer.build_matrix(m, 9)
        MtM = M.transpose().dot(M)
        MtS = M.transpose().dot(S)
        ans = np.array(scipy.sparse.linalg.spsolve(MtM, MtS))
        ans.shape = (9, k_features)


        sys.stdout.write("{:.2f}% - {:.2f}s \r".format((i/FACE_NUM)*100, time.time() - initial_time))
        sys.stdout.flush()

        return [ans, rfe.support_]

    # monta uma matrix 9*9*n_bodies
    def build_matrix(m_datas, basis_num):

        shape = (m_datas.shape[1] * basis_num, m_datas.shape[0] * basis_num)
        data = []
        rowid = []
        colid = []
        for i in range(0, m_datas.shape[1]):
            for j in range(0, basis_num):
                data += [c for c in m_datas[:, i].flat]
                rowid += [basis_num * i + j for a in range(m_datas.shape[0])]
                colid += [a for a in range(j * m_datas.shape[0], (j + 1) * m_datas.shape[0])]

        return scipy.sparse.coo_matrix((data, (rowid, colid)), shape)

    # calculating deform-based presentation(PCA)
    def get_d_basis(deformation, label="female"):

        print("Running Principal Component Analisys")

        mean_deform = deformation.mean(axis=0)
        mean_deform.shape = (FACE_NUM, 9)
        std_deform = deformation.std(axis=0)
        std_deform.shape = (FACE_NUM, 9)

        deformation -= mean_deform
        deformation /= std_deform
        deformation.shape = (deformation.shape[0], 9 * FACE_NUM)
        d = deformation.transpose()
        
        deformation_basis, d_sigma, V = np.linalg.svd(d, full_matrices=0)
        deformation_basis = np.array(deformation_basis[:, :D_BASIS_NUM]).reshape(9 * FACE_NUM, D_BASIS_NUM)
        deformation_coeff = np.dot(deformation_basis.transpose(), d)

        return deformation_basis, deformation_coeff

    def construct_coeff_mat(mat):
        return np.row_stack((-mat.sum(0), mat)).T

    def generate_vertex_deformation(d_inv_mean, facet, label="female"):
        print('Building vertex deformation matrix')
        data = []
        colidx = []
        off = VERTICES_NUM * 3
        
        shape = (FACE_NUM * 9, (VERTICES_NUM + FACE_NUM) * 3)

        rowidx = np.arange(shape[0])
        rowidx = np.repeat(rowidx, 4)

        for i in range(0, FACE_NUM):

            v = facet[i, :].flatten() - 1
            vector = np.hstack([v, i])
            vector = np.tile(vector, 3).reshape(3, vector.size) * 3
            vector = vector.T + np.array((0, 1, 2))
            vector = vector.T + np.array((0, 0, 0, off))
            vector = np.repeat(vector, 3, axis=0).flatten()
            colidx.extend(vector)

            coeff = Trainer.construct_coeff_mat(d_inv_mean[i])
            data.extend(np.tile(coeff.flatten(), 3))

        vertices_deformation = scipy.sparse.coo_matrix((data, (rowidx, colidx)), shape=shape)
        return vertices_deformation

    # calculate global mapping from measure to deformation PCA coeff
    def measure_to_deformation(d_coeff, t_measure):
        print("Build measure to deformation matrix")
        D = d_coeff.copy()
        D.shape = (D.size, 1)
        M = Trainer.build_matrix(t_measure, D_BASIS_NUM)
        MtM = M.transpose().dot(M)
        MtD = M.transpose().dot(D)
        ans = np.array(scipy.sparse.linalg.spsolve(MtM, MtD))
        ans.shape = (D_BASIS_NUM, MEASURE_NUM)
        return ans

if __name__ == "__main__":

    generated_files = os.listdir("./processed_data")

    if "cp.npz" in generated_files:
        with np.load("./processed_data/cp.npz", allow_pickle=True) as data:
            cp, vertices, faces, normals = data.values()
            cp = cp.item()
    else:
        cp, vertices, faces, normals = Loader.load()
        np.savez("./processed_data/cp.npz", cp, vertices, faces, normals)

    if "measure.npz" in generated_files:
        with np.load("./processed_data/measure.npz", allow_pickle=True) as data:
            measure, mean_measure, std_measure, normalized_measure = data.values()
    else:
        measure, mean_measure, std_measure, normalized_measure = Loader.load_measure_set(cp, vertices, faces)
        np.savez("./processed_data/measure.npz", measure, mean_measure, std_measure, normalized_measure)

    if "deformation.npz" in generated_files:
        with np.load("./processed_data/deformation.npz", allow_pickle=True) as data:
            deformation_inverse, deformation, determinants = data.values()

    else:
        deformation_inverse, deformation, determinants = Trainer.generate_face_deformation(vertices, faces)
        np.savez("./processed_data/deformation.npz", deformation_inverse, deformation, determinants)

    if "mask.npz" in generated_files:
        with np.load("./processed_data/mask.npz", allow_pickle=True) as data:
            mask, rfe_mat = data.values()
    else:
        mask, rfe_mat = Trainer.rfe_local(determinants, deformation, measure, normalized_measure)
        np.savez("./processed_data/mask.npz", mask, rfe_mat)


    if "deformation_basis.npz" in generated_files:
        with np.load("./processed_data/deformation_basis.npz", allow_pickle=True) as data:
            deformation_basis, deformation_coeff = data.values()
    else:
        deformation_basis, deformation_coeff = Trainer.get_d_basis(deformation)
        np.savez("./processed_data/deformation_basis.npz", deformation_basis, deformation_coeff)


    if "vertices_deformation.npy" in generated_files:
        vertices_deformation = np.load("./processed_data/vertices_deformation.npy", allow_pickle=True)
    else:
        vertices_deformation = Trainer.generate_vertex_deformation(deformation_inverse, faces)
        np.save("./processed_data/vertices_deformation.npy", vertices_deformation)


    if "measure_to_deformation.npy" in generated_files:
        measure_to_deformation = np.load("./processed_data/measure_to_deformation.npy", allow_pickle=True)
    else:
        measure_to_deformation = Trainer.measure_to_deformation(deformation_coeff, normalized_measure)
        np.save("./processed_data/measure_to_deformation.npy", measure_to_deformation)