import moderngl
import math
import predictor
import reshaper
import numpy as np
import time
import threading
from pyrr import Matrix44, matrix44
import moderngl_window as mglw
from tkinter import Tk, StringVar, W, E, S, N
from tkinter import ttk

TORADIANS = math.pi/180.0
MEASURE_NUM = 15


class ConfigWindow():

    def calculate(self, *args):
        try:
            value = float(self.feet.get())
            self.meters.set(int(0.3048 * value * 10000.0 + 0.5)/10000.0)
        except ValueError:
            pass

    def __init__(self):
        self.root = Tk()
        self.root.title("Feet to Meters")

        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.feet = StringVar()
        feet_entry = ttk.Entry(mainframe, width=7, textvariable=self.feet)
        feet_entry.grid(column=2, row=1, sticky=(W, E))

        self.meters = StringVar()
        ttk.Label(mainframe, textvariable=self.meters).grid(column=2, row=2, sticky=(W, E))

        ttk.Button(mainframe, text="Calculate", command=self.calculate).grid(column=3, row=3, sticky=W)

        ttk.Label(mainframe, text="feet").grid(column=3, row=1, sticky=W)
        ttk.Label(mainframe, text="is equivalent to").grid(column=1, row=2, sticky=E)
        ttk.Label(mainframe, text="meters").grid(column=3, row=2, sticky=W)

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

        feet_entry.focus()
        self.root.bind("<Return>", self.calculate)

    def update_loop(self):
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        self.root.quit()


class ViewWindow(mglw.WindowConfig):

    window_size = (1280, 720)
    # clear_color = None

    def close(self):
        self.config_window.close()

    def __init__(self, **kwargs):


        self.state = "render"
        self.normals = np.array([])
        self.vertices = np.array([])
        self.indices = np.array([])
        self.config_window = ConfigWindow()
        super().__init__(**kwargs)
        # Window & Context
        # self.ctx = moderngl.create_context()
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        data = np.full(MEASURE_NUM, np.nan)
        data[0] = 65
        data[1] = 165
        data[-1] = 19
        self.ager = predictor.Predictor(data)
        self.builder = reshaper.Reshaper()

        self.program = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 model;
                uniform mat4 view;
                uniform mat4 projection;
                uniform mat4 normal_matrix;
                in vec3 norm;
                in vec3 vert;
                out vec3 f_norm;
                out vec3 f_pos;
                void main() {
                    f_norm = vec3(normal_matrix*vec4(norm,1));
                    f_pos = vert;
                    gl_Position = projection * view * model * vec4(vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 frag_color;
                in vec3 f_norm;
                in vec3 f_pos;
                void main() {
                    vec3 lightPos = vec3(0,2,2);
                    vec3 lightColor = vec3(1,1,1);
                    vec3 vertex_normal = normalize(f_norm);
                    float ambient_strength = 0.50;
                    vec3 ambient = ambient_strength * lightColor;
                    vec3 lightDir = normalize(lightPos - f_pos);
                    float diff = max(dot(vertex_normal, lightDir), 0.0);
                    vec3 diffuse = diff * lightColor;
                    vec3 result = (diffuse+ambient) * vec3(0.5,0.5,0.5);
                    frag_color = vec4(result, 1.0);
                }
            ''',
        )

        self.model_uniform = self.program['model']
        self.view_uniform = self.program['view']
        self.projection_uniform = self.program['projection']
        self.normal_uniform = self.program['normal_matrix']

        self.projection = Matrix44.perspective_projection(60, 16.0/9.0, 0.1, 1000.0)
        self.view = Matrix44.look_at((0, 0, -2), (0, 0, 0), (0.0, 1.0, 0.0))   
    
    def render(self, time, frametime):

        try:
            self.config_window.update_loop()
        except Exception:
            exit(0)

        if self.state == "render":

            age_data = self.ager.predict_next(5)
            if self.ager.current_age > 95:
                self.state = "wait"
            else:
                self.vertices, self.normals, self.indices = self.builder.build_body(age_data.copy())

        else:
            self.close()
            exit(0)

        nbo = self.ctx.buffer(self.normals.astype('f4').tobytes())
        vbo = self.ctx.buffer(self.vertices.astype('f4').tobytes())
        ibo = self.ctx.buffer(self.indices.astype('i4').tobytes())

        content = [
            (nbo, '3f', 'norm'),
            (vbo, '3f', 'vert'),
        ]
        vao = self.ctx.vertex_array(self.program, content, ibo)

        model = Matrix44.from_translation((0, 0, 0), dtype=None)
        model = Matrix44.from_eulers((90*TORADIANS, 135*TORADIANS, 0))

        self.model_uniform.value = tuple(np.array(model).reshape(16))
        self.view_uniform.value = tuple(np.array(self.view).reshape(16))
        self.projection_uniform.value = tuple(np.array(self.projection).reshape(16))
        normal_matrix = matrix44.inverse(self.view * model).T
        self.normal_uniform.value = tuple(normal_matrix.reshape(16))

        vao.render(moderngl.TRIANGLES)


mglw.run_window_config(ViewWindow)
    
