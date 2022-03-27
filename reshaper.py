import scipy

class Reshaper:

    def __init__(self, loader, trainer):
        self.loader = loader
        self.trainer = trainer

    def build_body(self, measures):
        measures = measures.reshape(self.trainer.nmeasures, 1)
        measures *= self.trainer.std_measure
        measures += self.trainer.mean_measure

        d = []
        for i in range(0, self.trainer.nfaces):
            mask = self.trainer.mask_matrix[:, i]
            mask = mask.astype(bool)
            alpha = measures[mask]
            d.extend(self.trainer.deformation_matrix[i].dot(alpha))

        d = np.array(d)
        v, n, f = self.synthesize(d)
        return v, n, f

    # synthesize a body by deform-based, given deform, output vertex
    def synthesize(self, deform):
        AtD = self.trainer.vertex_deformations.transpose().dot(deform)
        vertices = self.trainer.lu.solve(AtD)
        vertices = vertices[:self.trainer.nvertices * 3].reshape(self.trainer.nvertices, 3)
        vertices -= np.mean(vertices, axis=0)
        normals = Reshaper.compute_normals(vertices, self.trainer.faces-1)
        return vertices, normals, self.trainer.faces-1


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
    print('File {} saved'.format(filename))

if __name__ == "__main__":

    from loader import Loader
    from trainer import Trainer
    from predictor import Predictor
    import numpy as np

    gender = "male"

    loader = Loader(gender=gender)
    faces, vertices, measures = loader.get_data()
    trainer = Trainer(faces, vertices, measures, gender=gender)

    data = np.full(16, np.nan)
    data[0] = 65.0
    data[1] = 165.0
    data[-1] = 19
    pred = Predictor(data, trainer, gender=gender)
    reshaper = Reshaper(loader, trainer)
    v, n, f = reshaper.build_body(pred.current_measures)
    save_obj("teste1.obj",v,f,n)