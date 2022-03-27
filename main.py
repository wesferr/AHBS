
import tkinter as tk

import math
import moderngl
import numpy as np
from loader import Loader
from predictor import Predictor
from reshaper import Reshaper
from trainer import Trainer
from pyrr import Matrix44, matrix44

from PIL import Image, ImageTk
TORADIANS = math.pi/180.0
size = (960, 720)


class RenderContext:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = "wait"

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

        self.projection = Matrix44.perspective_projection(60, 4.0/3.0, 0.1, 1000.0)
        self.view = Matrix44.look_at((0, 0, -2), (0, 0, 0), (0.0, 1.0, 0.0))

        self.rbo_rgba = ctx.renderbuffer(size)
        self.rbo_depth = ctx.depth_renderbuffer(size)
        self.fbo = ctx.framebuffer(self.rbo_rgba, self.rbo_depth)

    def setup_predictor(self, data, label='female'):
        self.loader = Loader(gender=label)
        faces, vertices, measures = self.loader.get_data()
        self.trainer = Trainer(faces, vertices, measures)
        self.builder = Reshaper(self.loader, self.trainer)
        self.ager = Predictor(data, self.trainer, gender=label)
        self.state = "render"

    def clear(self, color=(0, 0, 0, 0, 1.0)):
        self.ctx.clear(*color)

    def render(self):
        
        input_age_i = float(entries[-2].get()) if entries[-2].get() else 19.0
        input_age_f = float(entries[-1].get()) if entries[-1].get() else 80.0

        if self.state == "render":
          
            if input_age_i < input_age_f and self.ager.current_age < input_age_f:
                age_data = self.ager.predict_next(1)
                current_age_text.configure(text='Current Age: {}'.format(self.ager.current_age))
                self.vertices, self.normals, self.indices = self.builder.build_body(age_data.copy())
            elif input_age_i > input_age_f and self.ager.current_age > input_age_f:
                age_data = self.ager.predict_next(-1)
                current_age_text.configure(text='Current Age: {}'.format(self.ager.current_age))
                self.vertices, self.normals, self.indices = self.builder.build_body(age_data.copy())
            else:
                return


                
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

            self.fbo.use()
            self.fbo.clear(0.0, 0.0, 0.0, 1.0)
            vao.render(moderngl.TRIANGLES)
            image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
            reshaper.save_obj("obj/output/{}.obj".format(self.ager.current_age), self.vertices, self.indices, self.normals)
            image.save("photos/{}.png".format(self.ager.current_age))
            img.paste(image)

        else:
            pass


ctx = moderngl.create_context(require=430, standalone=True)
ctx.enable_only(moderngl.NOTHING)
ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
canvas = RenderContext(ctx)

root = tk.Tk()

img = ImageTk.PhotoImage(Image.new('RGB', size))
lbl = tk.Label(root, image=img)
lbl.grid(row=0, column=2, rowspan=18)
current_age_text = tk.Label(root, text='Current Age:')
current_age_text.grid(row=18, column=2, sticky=tk.N, padx=5)

inputs = [
    'Weight',
    'Height',
    'Waist Height',
    'Groin Height',
    'Arm Circumference',
    'Waist Circumference',
    'Arm Length',
    'Thigh Circumstance',
    'Shoulders Distance',
    'Calf Circumference',
    'Chest Circumference',
    'Neck Circumference',
    'Neck to Hip Distance',
    'Pulse Circumference',
    'Hip Circumference',
    '',
    'Start',
    'End',
]
entries = []

label = 'female'


def sel():
    global label
    if var.get() == 1:
        label = 'female'
    else:
        label = 'male'


var = tk.IntVar()
var.set(1)
R1 = tk.Radiobutton(root, text="Female", variable=var, value=1, command=sel)
R1.grid(row=0, column=0, sticky=tk.W, padx=5)

R2 = tk.Radiobutton(root, text="Male", variable=var, value=2, command=sel)
R2.grid(row=0, column=1, sticky=tk.W, padx=5)

for index, value in enumerate(inputs):
    if index != 15:
        text_value = '{}:'.format(value)
        text = tk.Label(root, text=text_value)
        text.grid(row=index+1, column=0, sticky=tk.W, padx=5)
        metric = tk.Entry(root)
        metric.grid(row=index+1, column=1)
        entries.append(metric)

text = tk.Label(root, text='Age')
text.grid(row=16, column=0, columnspan=2)

running = True


def update():
    data_change_handler()
    canvas.clear()
    canvas.render()


def event_handler(event="X"):
    if event == "X" or (isinstance(event, tk.Event) and event.keysym == "Escape"):
        global running
        running = False


def data_change_handler():
    if hasattr(canvas, 'ager'):
        data = canvas.ager.get_denormalized_current_measures()
        for index, entry in enumerate(entries[:-2]):
            entry.delete(0, len(entry.get()))
            entry.insert(0, np.round(data[index], 6))
    else:
        #  not ready yet
        pass


def button_handler():
    data = [entry.get() for entry in entries]
    data = np.array([float(i) if i.isnumeric() else np.nan for i in data])
    canvas.setup_predictor(data[:-1], label=label)


btn = tk.Button(root, text='Simulate', command=button_handler)
btn.grid(row=19, column=0, columnspan=2)

root.protocol("WM_DELETE_WINDOW", event_handler)
root.bind('<Escape>', event_handler)

while running:
    root.update_idletasks()
    update()
    root.update()




# from trainer import Trainer
# from loader import Loader

# if __name__ == "__main__":
#     loader = Loader(gender="female")
#     faces, vertices, measures = loader.get_data()
#     Trainer(faces, vertices, measures)
