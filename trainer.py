import numpy as np
import scipy
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from tqdm import tqdm as progressbar


class Trainer:
    def __init__(self, faces, vertices, measures, gender="female", nfeatures=9):

        self.vertices = vertices
        self.faces = faces
        self.measures = measures
        self.vertices_mean = self.vertices.mean(axis=0)

        self.nfeatures = nfeatures
        self.nfaces = self.faces.shape[0]
        self.nmeasures = self.measures.shape[0]
        self.nbodies = self.measures.shape[1]
        self.nvertices = self.vertices.shape[1]

        self.mean_measure = measures.mean(axis=1).reshape(self.nmeasures, 1)
        self.std_measure = measures.std(axis=1).reshape(self.nmeasures, 1)
        self.normalized_measures = self.measures - self.mean_measure
        self.normalized_measures /= self.std_measure

        try:
            self.determinants = np.load("processed_data/{}_determinants.npy".format(gender))
            self.deformation = np.load("processed_data/{}_deformation.npy".format(gender))
            self.deformation_inverse = np.load("processed_data/{}_deformation_inverse.npy".format(gender))
        except:
            values = self.generate_face_deformation()
            self.determinants = values[0]
            self.deformation = values[1]
            self.deformation_inverse = values[2]
            np.save("processed_data/{}_determinants.npy".format(gender), self.determinants)
            np.save("processed_data/{}_deformation.npy".format(gender), self.deformation)
            np.save("processed_data/{}_deformation_inverse.npy".format(gender), self.deformation_inverse)


        try:
            self.deformation_matrix = np.load("processed_data/{}_deformation_matrix.npy".format(gender))
            self.mask_matrix = np.load("processed_data/{}_mask_matrix.npy".format(gender))
        except:
            values = self.recursive_elimination()
            self.deformation_matrix = values[0]
            self.mask_matrix = values[1]
            np.save("processed_data/{}_deformation_matrix.npy".format(gender), self.deformation_matrix)
            np.save("processed_data/{}_mask_matrix.npy".format(gender), self.mask_matrix)

        try:
            self.deformation_matrix = np.load("processed_data/{}_vertex_deformations.npy".format(gender))
        except:
            values = self.generate_vertices_deformation()
            self.vertex_deformations = values
            np.save("processed_data/{}_vertex_deformations.npy".format(gender), self.vertex_deformations)

        self.lu = scipy.sparse.linalg.splu(
            self.vertex_deformations.transpose().dot(self.vertex_deformations)
        )


    """  BODY DEFORMATION GENERATION SECTION  """

    def generate_face_deformation(self):
        deform_inverse = self.calculate_deformation_inverse()
        deform, determ = self.calculate_deformation(deform_inverse)
        return determ, deform, deform_inverse

    def calculate_deformation_inverse(self):
        inverse_deformation = np.zeros((self.nfaces, 3, 3))
        faces_list = range(self.nfaces)
        for idx in progressbar(faces_list, desc="Faces Inverse Deformation"):
            face = self.faces[idx] - 1
            v_1 = self.vertices_mean[face[0]]
            v_2 = self.vertices_mean[face[1]]
            v_3 = self.vertices_mean[face[2]]
            deformation = Trainer.calculate_face_deformation(v_1, v_2, v_3)
            inverse_deformation[idx] = np.linalg.inv(deformation)
        return inverse_deformation

    def calculate_deformation(self, deformation_inverse):

        # deformation_inverse is V-1 in sumner article
        # face_deformation is V~ in sumner article
        # source_face_deformation is Q in sumner article
        # determinants is Y matrix of Zeng article

        deform = np.zeros((self.nbodies, self.nfaces, 9))
        determ = []

        ntasks = 50
        tasks = [(
            self.faces[face_id:face_id+ntasks],
            self.nbodies,
            self.vertices,
            deformation_inverse[face_id:face_id+ntasks],
        ) for face_id in range(0, self.nfaces, ntasks)]

        pool = Pool(processes=cpu_count())
        results = list(
            progressbar(
                pool.imap(Trainer.parallel_calculate_deformation_map, tasks),
                total=len(tasks),
                desc="Faces Deformation"
            )
        )
        pool.close()
        pool.join()

        deform = np.array([result[0] for result in results])
        determ = np.array([result[1] for result in results])
            
        deform = deform.reshape(self.nbodies, self.nfaces, 9)
        determ = determ.reshape(self.nfaces, self.nbodies)

        return deform, determ

    def parallel_calculate_deformation_map(args):

        return Trainer.parallel_calculate_deformation(*args)

    def parallel_calculate_deformation(faces, nbodies, vertices, deformations_inverse):
        import gc
        deform = []
        determ = []
        for i in range(len(faces)):
            face = faces[i] - 1
            deformation_inverse = deformations_inverse[i]
            for vertice_id in range(nbodies):
                vertice_1 = vertices[vertice_id, face[0]]
                vertice_2 = vertices[vertice_id, face[1]]
                vertice_3 = vertices[vertice_id, face[2]]
                face_deform = Trainer.calculate_face_deformation(
                    vertice_1, vertice_2, vertice_3
                )
                source_deform = face_deform.dot(deformation_inverse)
                determ.append(np.linalg.det(source_deform))
                deform.append(source_deform.flatten())

        return deform, determ

    def calculate_face_deformation(v_1, v_2, v_3):
        v_21 = v_2 - v_1
        v_31 = v_3 - v_1
        v_41 = np.cross(v_21, v_31)
        v_41 /= np.sqrt(np.linalg.norm(v_41))
        v31_v41 = np.column_stack((v_31, v_41))
        face_deformation = np.column_stack((v_21, v31_v41))
        return face_deformation


    """  RECURSIVE FEATURE ELIMINATION SECTION  """

    def recursive_elimination(self):
        tasks = [
            (
                self.determinants[i, :],
                self.deformation[:, i, :],
                self.normalized_measures,
                self.nfeatures,
                self.nmeasures,
                self.nbodies,
                self.measures,
            )
            for i in range(self.nfaces)
        ]
        # results = pool.starmap(Trainer.parallel_measure_elimination, progressbar(tasks, total=len(tasks) ,desc="Recursive Feature Elimination"))

        pool = Pool(processes=cpu_count())
        results = list(
            progressbar(
                pool.imap(Trainer.parallel_measure_elimination_map, tasks),
                total=len(tasks),
                desc="Recursive Feature Elimination"
            )
        )
        pool.close()
        pool.join()

        deformation_matrix = np.array([])
        mask_matrix = np.array([])
        for result in results:
            deformation_matrix = np.append(deformation_matrix, result[0])
            mask_matrix = np.append(mask_matrix, result[1])

        deformation_matrix = np.array(deformation_matrix).reshape(
            self.nfaces, 9, self.nfeatures
        )
        mask_matrix = mask_matrix.reshape(self.nfaces, self.nmeasures)
        mask_matrix = mask_matrix.transpose()

        return deformation_matrix, mask_matrix

    # deformation_list is de S on sumner article
    # selected_measures is de M on zeng article
    # ans is the P´ on zeng article

    def parallel_measure_elimination_map(args):
        return Trainer.parallel_measure_elimination(*args)

    def parallel_measure_elimination(determinants, deformation,
        normalized_measures, nfeatures, nmeasures, nbodies, measures):

        x = normalized_measures.transpose()
        LR = LinearRegression()
        recursion_manager = RFE(LR, n_features_to_select=nfeatures)
        recursion_manager.fit(x, determinants)

        mask = recursion_manager.get_support().reshape(nmeasures, 1)
        bodies_mask = mask.repeat(nbodies, axis=1)

        selected_measures = measures[bodies_mask]
        selected_measures.shape = (nfeatures, nbodies)
        measures_matrix = Trainer.build_measure_matrix(selected_measures, nbodies, nfeatures)
        deformation_list = deformation.reshape(deformation.size, 1)

        MtM = measures_matrix.transpose().dot(measures_matrix)
        MtD = measures_matrix.transpose().dot(deformation_list)
        ans = scipy.sparse.linalg.spsolve(MtM, MtD)
        return ans, mask

    def build_measure_matrix(selected_measures, nbodies, nfeatures):
        results = []
        for body in range(nbodies):
            for i in range(9):
                M = np.zeros((9, nfeatures))
                M[i] = selected_measures[:, body]
                results.append(M.flatten())
        return scipy.sparse.csc_matrix(results)


    """  VERTICES DEFORMATION GENERATION SECTION  """

    def generate_vertices_deformation(self):
        data = []
        colidx = []
        off = self.nvertices * 3
        
        shape = (self.nfaces * 9, (self.nvertices + self.nfaces) * 3)

        rowidx =  np.arange(shape[0])
        rowidx = np.repeat(rowidx, 4)

        faces_list = range(self.nfaces)
        for face_id in progressbar(faces_list, desc="Vertices Deformation Matrix"):
            v = self.faces[face_id, :].flatten() - 1
            vector = np.hstack([v, face_id])
            vector = np.tile(vector, 3).reshape(3, vector.size) * 3
            vector = vector.T + np.array((0, 1, 2))
            vector = vector.T + np.array((0, 0, 0, off))
            vector = np.repeat(vector, 3, axis=0).flatten()
            colidx.extend(vector)

            coeff = self.construct_coeff_mat(self.deformation_inverse[face_id])
            data.extend(np.tile(coeff.flatten(), 3))

        structure = (data, (rowidx, colidx))
        vertices_deformation = scipy.sparse.coo_matrix(structure, shape=shape)
        return vertices_deformation

    def construct_coeff_mat(self, mat):
        return np.row_stack((-mat.sum(0), mat)).T
