import numpy as np
import scipy as scp
import time
import os
import sys

FACE_NUM = 25000
VERTICES_NUM = 12500
MEASURE_NUM = 14


class Loader:

    def load():
        cp = Loader.load_control_points("base_data/control_points.npz")
        faces, normals = Loader.load_template("base_data/template.obj")
        vertex = Loader.load_models("obj/", label='female')
        return cp, vertex, faces, normals

    def load_control_points(path):
        print('Loading control points')
        metricas = {}
        with np.load(path) as file:
            for metrica in file.files:
                metricas[metrica] = file[metrica]
        return metricas

    # load facet information from txt file
    def load_template(path):
        print("Loading template")
        facet = np.zeros((FACE_NUM, 3), dtype=int)
        normals = np.zeros((FACE_NUM, 3), dtype=int)
        f = open(path, 'r')
        i = 0
        for line in f:
            if line[0] == 'f':
                line_split = line[1:].split()
                line_split = np.array([i.split("//") for i in line_split])
                face_temp = list(map(int, line_split[:, 0]))
                normal_temp = list(map(int, line_split[:, 1]))
                facet[i, :] = face_temp
                normals[i, :] = normal_temp
                i += 1
        return facet, normals

    # loading data: file_list, vertex, mean, std
    def load_models(path, label="female"):
        print('Loading {} models'.format(label))
        obj_file_dir = os.path.join(path, label)
        file_list = os.listdir(obj_file_dir)

        vertex = []
        for i, obj in enumerate(file_list):
            f = open(os.path.join(obj_file_dir, obj), 'r')
            for line in f:
                if line[0] == '#':
                    continue
                elif "v " in line:
                    line.replace('\n', ' ')
                    tmp = list(map(float, line[1:].split()))
                    vertex.extend(tmp)
                else:
                    break
        file_list_size = len(file_list)
        vertex = np.array(vertex).reshape(file_list_size, VERTICES_NUM, 3)
        return vertex

    # calculate measure data from given vertex by control points
    def load_measures(cp, vertex, facet):
        measure_list = []
        # clac weight
        vol = 0.0
        kHumanbodyIntensity = 1026.0
        for i in range(0, FACE_NUM):
            f = [c - 1 for c in facet[i, :]]
            v0 = vertex[f[0], :]
            v1 = vertex[f[1], :]
            v2 = vertex[f[2], :]
            vol += np.cross(v0, v1).dot(v2)
        vol = abs(vol) / 6.0
        weight = kHumanbodyIntensity * vol
        weight = weight**(1.0 / 3.0) * 1000
        measure_list.append(weight)

        # calc height
        up = vertex[275]
        down = vertex[12478]
        height = abs(down[2]-up[2])
        measure_list.append(height * 1000)

        # calc other measures
        for measure in cp:
            point_list = cp[measure]
            length = 0.0
            p2 = (vertex[int(point_list[0][0]), :] * point_list[0][3] +
                  vertex[int(point_list[0][1]), :] * point_list[0][4] +
                  vertex[int(point_list[0][2]), :] * point_list[0][5])
            for point in point_list[1:]:
                p1 = p2
                p2 = (vertex[int(point[0]), :] * point[3] +
                      vertex[int(point[1]), :] * point[4] +
                      vertex[int(point[2]), :] * point[5])
                length += np.sqrt(np.sum((p1 - p2)**2.0))
            measure_list.append(length * 1000)
        return np.array(measure_list).reshape(MEASURE_NUM, 1)

    def load_measure_set(cp, vertex, facet):
        print('Loading measurements data')
        measure = np.zeros((MEASURE_NUM, vertex.shape[0]))

        initial_time = time.time()
        for i in range(vertex.shape[0]):
            sys.stdout.write("{:.2f}% - {:.2f}s \r".format((i/vertex.shape[0])*100, time.time() - initial_time))
            sys.stdout.flush()
            measure[:, i] = Loader.load_measures(cp, vertex[i, :, :], facet).flat

        mean_measure = np.array(measure.mean(axis=1)).reshape(MEASURE_NUM, 1)
        std_measure = np.array(measure.std(axis=1)).reshape(MEASURE_NUM, 1)
        normalized_measure = measure - mean_measure
        normalized_measure /= std_measure
        return measure, mean_measure, std_measure, normalized_measure


if __name__ == "__main__":
    cp, vertices, faces, normals = Loader.load()
    measure, mean_measure, std_measure, normalized_measure = Loader.load_measure_set(cp, vertices, faces)
