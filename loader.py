import numpy as np
import os
from tqdm import tqdm as progressbar


class Loader:
    def __init__(self, gender="female"):

        self.gender = gender

        self.control_points_path = "base_data/control_points.npz"
        self.body_template_path = "base_data/template.obj"
        self.bodies_files_path = "bodies/"

        self.control_points = self.load_control_points()
        self.nmeasures = len(self.control_points) + 2

        self.faces, self.normals = self.load_body_template()
        self.nfaces = self.faces.shape[0]

        try:
            self.bodies = np.load("processed_data/{}_bodies.npy".format(gender))
        except:
            self.bodies = self.load_body_models()
            np.save("processed_data/{}_bodies.npy".format(gender), self.bodies)
        self.nbodies = self.bodies.shape[0]

        try:
            self.measures_set = np.load("processed_data/{}_measure_set.npy".format(gender))
        except:
            self.measures_set = self.load_measures_set()
            np.save("processed_data/{}_measure_set.npy".format(gender), self.measures_set)

        

    def get_data(self):
        return self.faces, self.bodies, self.measures_set

    def load_control_points(self):
        metricas = {}
        with np.load(self.control_points_path) as file:
            for metrica in file.files:
                metricas[metrica] = file[metrica]
        return metricas

    def load_body_template(self):
        faces = []
        normals = []
        i = 0
        with open(self.body_template_path) as file:
            for line in file:
                if line[0] == "f":
                    line_split = line[1:].split()
                    line_split = np.array([i.split("//") for i in line_split])
                    face_temp = list(map(int, line_split[:, 0]))
                    normal_temp = list(map(int, line_split[:, 1]))
                    faces.append(face_temp)
                    normals.append(normal_temp)
                    i += 1
        return np.array(faces), np.array(normals)

    def load_body_models(self):
        obj_file_dir = os.path.join(self.bodies_files_path, self.gender)
        file_list = os.listdir(obj_file_dir)
        bodies = []
        nvertices = 0

        for obj in progressbar(file_list, desc='Loading Bodies'):
            with open(os.path.join(obj_file_dir, obj), "r") as file:
                vertices = []
                for line in file:
                    if line[0] == "#":
                        continue
                    elif "v " in line:
                        line.replace("\n", " ")
                        tmp = list(map(float, line[1:].split()))
                        vertices.append(tmp)
                    else:
                        break
                nvertices = len(vertices)
                bodies.append(vertices)
        bodies = np.array(bodies).reshape(len(bodies), nvertices, 3)
        return bodies

    # calculate measure data from given vertex by control points
    def load_measures(self, vertex):
        measure_list = []

        # vol = 0.0
        # kHumanbodyIntensity = 1026.0
        # for i in range(0, self.nfaces):
        #     f = [c - 1 for c in self.faces[i, :]]
        #     v0 = vertex[f[0], :]
        #     v1 = vertex[f[1], :]
        #     v2 = vertex[f[2], :]
        #     vol += np.cross(v0, v1).dot(v2)
        # vol = abs(vol) / 6.0
        # weight = kHumanbodyIntensity * vol
        # weight = weight**(1.0 / 3.0) * 1000
        # measure_list.append(weight)

        # calculate height
        up = vertex[275]
        down = vertex[12478]
        height = abs(down[2] - up[2]) * 1000
        measure_list.append(height)

        # calculate other measures
        for measure in self.control_points:
            point_list = self.control_points[measure]
            length = 0.0
            p2 = (
                vertex[int(point_list[0][0]), :] * point_list[0][3]
                + vertex[int(point_list[0][1]), :] * point_list[0][4]
                + vertex[int(point_list[0][2]), :] * point_list[0][5]
            )
            for point in point_list[1:]:
                p1 = p2
                p2 = (
                    vertex[int(point[0]), :] * point[3]
                    + vertex[int(point[1]), :] * point[4]
                    + vertex[int(point[2]), :] * point[5]
                )
                length += np.sqrt(np.sum((p1 - p2) ** 2.0))
            measure_list.append(length * 1000)

        # print(measure_list[3]/10, measure_list[4]/10, measure_list[8]/10)
        estimated_weight = (
            (0.5759 * measure_list[3]/10)
            + (0.5263 * measure_list[4]/10)
            + (1.2452 * measure_list[8]/10)
            - (4.8689 * (1 if self.gender == "male" else 2))
            - 32.9241
        )  # doi: 10.1590/1980-0037.2014v16n4p475

        estimated_weight = estimated_weight**(1.0 / 3.0) * 1000
        measure_list = [estimated_weight] + measure_list

        return np.array(measure_list).reshape(self.nmeasures, 1)

    def load_measures_set(self):
        measures_set = np.zeros((self.nmeasures, self.nbodies))
        bodies_list = range(self.nbodies)
        for i in progressbar(bodies_list, desc='Loading Measure Set'):
            measures_set[:, i] = self.load_measures(self.bodies[i, :, :]).flat
        return measures_set


if __name__ == "__main__":
    # loader = Loader(gender="male")
    loader = Loader(gender="female")
