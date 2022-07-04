import numpy as np
from pyrfc3339 import generate

class CurveUtils():

    def load_template(file_path):
        faces = []
        normals = []
        i = 0
        with open(file_path) as file:
            for line in file:
                if line[0] == "f":
                    line_split = line[1:].split()
                    line_split = np.array([i.split("//") for i in line_split])

                    face_temp = list(map(int, line_split[:, 0]))
                    faces.append(face_temp)

                    if line_split[0].size > 1:
                        normal_temp = list(map(int, line_split[:, 1]))
                        normals.append(normal_temp)

                    i += 1
        return np.array(faces)-1, np.array(normals)

    def load_mesh(file_path):
        vertex = []
        with open(file_path, "r") as f:
            for line in f:
                if line[0] == '#':
                    continue
                elif "v " in line:
                    line.replace('\n', ' ')
                    tmp = list(map(float, line[1:].split()))
                    vertex.append(tmp)
                else:
                    break
        return np.array(vertex)

    def save_obj(path, pontos, faces=[]):
        with open(path, "w") as file:
            for ponto in pontos:
                file.write("v {} {} {}\n".format(ponto[0], ponto[1], ponto[2]))
            for face in faces:
                file.write("f {} {} {}\n".format(face[0], face[1], face[2]))

    def generate_positions(coordinates, vertices):
        positions = []
        for i in range(0, len(coordinates)):
            x = vertices[int(coordinates[i][0])] * coordinates[i][3]
            y = vertices[int(coordinates[i][1])] * coordinates[i][4]
            z = vertices[int(coordinates[i][2])] * coordinates[i][5]
            p = x + y + z
            positions.append(p)
        return np.array(positions)

    def calculate_length(coordinates, vertices):
        length = 0.0
        p2 = (
            vertices[int(coordinates[0][0]), :] * coordinates[0][3]
            + vertices[int(coordinates[0][1]), :] * coordinates[0][4]
            + vertices[int(coordinates[0][2]), :] * coordinates[0][5]
        )
        for point in coordinates[1:]:
            p1 = p2
            p2 = (
                vertices[int(point[0]), :] * point[3]
                + vertices[int(point[1]), :] * point[4]
                + vertices[int(point[2]), :] * point[5]
            )
            length += np.sqrt(np.sum((p1 - p2) ** 2.0))
        return length * 100

    def calculate_height(floor, coordiantes, body):
        positions = CurveUtils.generate_positions(coordiantes, body)
        height = abs(floor - positions.mean(axis=0)[1])
        return height * 100
