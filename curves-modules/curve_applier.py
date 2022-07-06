import numpy as np
from curve_utils import CurveUtils
from star_models.star import STAR
import tensorflow as tf 
from tqdm import tqdm as progressbar

class CurveApplier():
    
    def __init__(self, medidas_pose_0, curves_0) -> None:
        self.medidas_pose_0 = medidas_pose_0
        self.curves_0 = curves_0

    def generate_measures(self, gender="female"):
        bodies_measures = []
        bodies_0 = np.load(f"bodies-{gender}-pose-0.npy")
        bodies_1 = np.load(f"bodies-{gender}-pose-1.npy")
        for idx in progressbar(range(bodies_0.shape[0])):

            body = bodies_0[idx]
            body_1 = bodies_1[idx]

            measures = []
            floor = body[:,1].min()
            top = body[:,1].max()
            measures.append(abs(floor-top)*100) #stature

            for medida in self.medidas_pose_0:
                length = CurveUtils.calculate_length(self.curves_0[medida[2]], body)
                measures.append(length) # circunferences

            curve = self.curves_0['shoulder-circunference']
            shoulder_positions = CurveUtils.generate_positions(curve, body_1)
            dimensions = abs(shoulder_positions.min(axis=0) - shoulder_positions.max(axis=0))
            dimension = np.array(dimensions).argmax()
            length = abs(shoulder_positions[:,dimension].min() - shoulder_positions[:,dimension].max())
            measures.append(length*100)


            length = CurveUtils.calculate_length(self.curves_0['neck_to_waist'], body)
            measures.append(length)

            height = CurveUtils.calculate_height(floor, self.curves_0['waist-girth'], body)
            measures.append(height)

            height = CurveUtils.calculate_height(floor, self.curves_0['hip-girth'], body)
            measures.append(height)

            bodies_measures.append(measures)

        bodies_measures = np.array(bodies_measures)
        np.save(f'bodies-{gender}-measures.npy', bodies_measures)

def generate_star_body(star, pose, betas, trans):
    vertices = star(pose,betas,trans)[0]
    return vertices

def generate_bodies(gender='female', nbodies=3000):

    batch_size = 3
    star = STAR(gender=gender, num_betas=batch_size)

    bodies_pose_0 = []
    bodies_pose_1 = []
    for x in progressbar(range(nbodies)):

        trans = np.zeros((1,3))
        trans = tf.constant(trans, dtype=tf.float32)

        betas = (np.random.rand(*(1,batch_size)) - 0.5) * 6
        betas = tf.constant(betas, dtype=tf.float32)

        # pose 0
        pose = np.zeros((1,72))
        pose = tf.constant(pose, dtype=tf.float32)
        body = generate_star_body(star, pose, betas, trans)
        bodies_pose_0.append(body)

        # pose 1
        pose = np.zeros((1,72))
        pose[0][53] = 80 * np.pi/180
        pose[0][50] = -80 * np.pi/180
        pose = tf.constant(pose, dtype=tf.float32)
        body = generate_star_body(star, pose, betas, trans)
        bodies_pose_1.append(body)


    np.save(f"bodies-{gender}-pose-0.npy", np.array(bodies_pose_0))
    np.save(f"bodies-{gender}-pose-1.npy", np.array(bodies_pose_1))

def generate_mean_body(gender='female'):

    batch_size = 3
    star = STAR(gender=gender, num_betas=batch_size)

    trans = tf.constant(np.zeros((1,3)), dtype=tf.float32)
    betas = tf.constant(np.zeros((1,batch_size)), dtype=tf.float32)

    # pose 0
    pose = tf.constant(np.zeros((1,72)), dtype=tf.float32)
    body1 = generate_star_body(star, pose, betas, trans)

    # pose 1
    pose = np.zeros((1,72))
    pose[0][53] = 80 * np.pi/180
    pose[0][50] = -80 * np.pi/180
    pose = tf.constant(pose, dtype=tf.float32)
    body2 = generate_star_body(star, pose, betas, trans)

    return body1, body2