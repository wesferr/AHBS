import numpy as np
from curve_utils import CurveUtils
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.spatial.distance import cdist

class CurveGenerator():

    def __init__(self, mesh_path, template_path) -> None:
        self.nplano = np.array([])
        self.dplano = None
        self.vertice_coordinates = np.array([])
        self.hashmap = {}

        self.axis_dict = { "x": 0, "y": 1, "z": 2 }
        self.axis_planes = {
            "x": lambda i : [ [ i,  3,  3], [ i, -3,  3], [ i,  3, -3], [ i, -3, -3] ], 
            "y": lambda i : [ [ 3,  i,  3], [-3,  i,  3], [ 3,  i, -3], [-3,  i, -3] ], 
            "z": lambda i : [ [ 3,  3,  i], [-3,  3,  i], [ 3, -3,  i], [-3, -3,  i] ]
        }

        self.vertex = CurveUtils.load_mesh(mesh_path)
        self.faces, self.normals = CurveUtils.load_template(template_path)

    def set_plane(self, p1, p2, p3, p4):
        self.pvertices = np.array([p1,p2,p3,p4])
        v1 = self.pvertices[0] - self.pvertices[1]
        v2 = self.pvertices[0] - self.pvertices[2]

        cross = np.cross(v1,v2)
        self.nplano = cross/np.linalg.norm(cross)
        self.dplano = self.nplano.dot(-self.pvertices[0])

    def calculate_plane_intersection(self, vertice):
        ax = self.nplano[0] * vertice[0]
        by = self.nplano[1] * vertice[1]
        cz = self.nplano[2] * vertice[2]
        return ax+by+cz+self.dplano

    def inside_triangle(self, a,b,c):
        if a > 0 and b > 0 and c > 0:
            return False
        elif a < 0 and b < 0 and c < 0:
            return False
        else:
            return True

    def outside_plane(self, vertice):
        p0 = self.pvertices[0]
        p1 = self.pvertices[1]
        p2 = self.pvertices[2]
        p3 = self.pvertices[3]
        value1 = self.baricentric_coordinates(p0,p1,p2, vertice)
        value2 = self.baricentric_coordinates(p1,p2,p3, vertice)
        return min(value1) < 0 and min(value2) < 0

    def calculate_intersection_point(self, p0, p1):
        vd = p1 - p0
        integer_part = self.nplano.dot(p0)
        incognita_part = self.nplano.dot(vd)
        t = (self.dplano + integer_part)/incognita_part
        intersection_point = p0 + (-t*vd)
        return intersection_point

    def add_to_coordinates(self, v1, v2, v3, i1, i2, i3):

        point = self.calculate_intersection_point(v2,v1)
        if self.outside_plane(point): return

        array = np.array([(i1,i2,i3), self.baricentric_coordinates(v1, v2, v3, point)])
        
        hashfunc = (i1*i2)+i1+i2
        if hashfunc not in self.hashmap:
            self.hashmap[hashfunc] = True
            self.vertice_coordinates.append(array.reshape(array.size))

    def baricentric_coordinates(self, a,b,c,p):
        vab = b - a
        vbc = c - b
        vca = a - c
        
        vap = p - a
        vbp = p - b
        vcp = p - c
        
        n = np.cross(vab,vbc) / np.linalg.norm(np.cross(vab,vbc))
        
        ABC = np.dot(n, np.cross(vab, vbc)) / 2
        ABP = np.dot(n, np.cross(vab, vbp)) / 2
        BCP = np.dot(n, np.cross(vbc, vcp)) / 2
        CAP = np.dot(n, np.cross(vca, vap)) / 2
        
        w = ABP/ABC
        u = BCP/ABC
        v = CAP/ABC
        
        return np.array([u, v, w])

    def generate_base_curve(self):
        assert self.nplano.size != 0,"Plane not defined"
        assert self.dplano != None,"Plane not defined"

        self.vertice_coordinates = []
        self.hashmap = {}

        for face in self.faces:
            i1 = face[0]
            v1 = self.vertex[i1]
            r1 = self.calculate_plane_intersection(v1)
            
            i2 = face[1]
            v2 = self.vertex[i2]
            r2 = self.calculate_plane_intersection(v2)

            i3 = face[2]
            v3 = self.vertex[i3]
            r3 = self.calculate_plane_intersection(v3)
            

            result = self.inside_triangle(r1,r2,r3)
            if result:
                if r2 * r3 > 0:
                    self.add_to_coordinates(v1, v2, v3, i1, i2, i3)
                    self.add_to_coordinates(v1, v3, v2, i1, i3, i2)

                elif r1 * r3 > 0:
                    self.add_to_coordinates(v2, v1, v3, i2, i1, i3)
                    self.add_to_coordinates(v2, v3, v1, i2, i3, i1)
                    
                elif r1 * r2 > 0:
                    self.add_to_coordinates(v3, v2, v1, i3, i2, i1)
                    self.add_to_coordinates(v3, v1, v2, i3, i1, i2)
            
        self.vertice_coordinates = np.array(self.vertice_coordinates)

    def get_curve_structure(self):
        return self.vertice_coordinates

    def sort_as_curve(self):

        positions = CurveUtils.generate_positions(self.vertice_coordinates, self.vertex)

        distances = cdist(positions, positions)
        data = distances*3e9
        data = (data).astype(int)
        
        def get_solution(manager, routing, solution):
            order = []
            index = routing.Start(0)
            order.append(index)
            while not routing.IsEnd(index):
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                order.append(index)
            return order

        def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return data[from_node][to_node]

        manager = pywrapcp.RoutingIndexManager(len(distances), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        solution = routing.SolveWithParameters(search_parameters)

        permutation = get_solution(manager, routing, solution)[:-1]
        
        coordinates=[]
        for i in permutation:
            coordinates.append(self.vertice_coordinates[i])

        self.vertice_coordinates = np.array(coordinates)

    def sort_as_line(self):
        positions = CurveUtils.generate_positions(self.vertice_coordinates, self.vertex)
        sort_range = positions[:, 1].argsort()
        self.vertice_coordinates = self.vertice_coordinates[sort_range]
        return self.vertice_coordinates

    def generate_curve(self, medida, npy_output):
        plane = self.axis_planes[medida[0]](medida[1])
        self.set_plane(*plane)
        self.generate_base_curve()
        self.sort_as_curve()
        np.save("{}{}.npy".format(npy_output, medida[2]), self.vertice_coordinates)
        return self.vertice_coordinates

    def generate_line(self, plane, name, npy_output):
        self.set_plane(*plane)
        self.generate_base_curve()
        self.sort_as_line()
        np.save("{}{}.npy".format(npy_output, name), self.vertice_coordinates)
        return self.vertice_coordinates

    def get_body(self):
        return self.vertex, self.faces, self.normals
