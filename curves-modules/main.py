import argparse
import numpy as np
from scipy import rand
from tqdm import tqdm as progressbar
from curve_generator import CurveGenerator
from curve_applier import CurveApplier, generate_bodies, generate_mean_body
from curve_utils import CurveUtils

def generate_measures(pose_0, pose_1, male_body):

    medidas = [
        ('y', -0.129292, 'waist-girth'),
        ('y', -0.277272, 'hip-girth'),
        ('y', -0.345454, 'thigh-girth'),
        ('y', -0.830303, 'calf-girth'),
        ('y',  0.293333, 'neck-girth'),
        ('y',  0.066464, 'bust-girth'),
        ('x', -0.660000, 'wrist-girth'),
        ('x', -0.298484, 'upper-arm-girth'),
    ]

    curves = {}
    positions = []

    for medida in progressbar(medidas):
        curve = pose_0.generate_curve(medida, './npy-output/')
        positions.append(CurveUtils.generate_positions(curve, male_body))
        curves[medida[2]] = curve

    medida = ('y',  0.275200, 'shoulder-circunference')
    curve = pose_1.generate_curve(medida, './npy-output/')
    positions.append(CurveUtils.generate_positions(curve, male_body))
    curves['shoulder-circunference'] = curve

    # neck to waist
    neck_positions = CurveUtils.generate_positions(curves['neck-girth'], pose_0.get_body()[0])
    waist_positions = CurveUtils.generate_positions(curves['waist-girth'], pose_0.get_body()[0])
    neck_height = neck_positions.mean(axis=0)[1]
    waist_height = waist_positions.mean(axis=0)[1]
    plane = [ [ 0,  neck_height,  0], [ 0, waist_height,  0], [ 0,  neck_height, -3], [ 0, waist_height, -3] ]
    curve = pose_0.generate_line(plane, 'neck_to_waist', './npy-output/')
    positions.append(CurveUtils.generate_positions(curve, male_body))
    curves['neck_to_waist'] = curve

    return medidas, curves, positions


if __name__ == '__main__':

    pose_0 = CurveGenerator('mean_female_pose_0.obj', 'mean_female_pose_0.obj')
    pose_1 = CurveGenerator('mean_female_pose_1.obj', 'mean_female_pose_1.obj')

    male_mesh = CurveUtils.load_mesh('mean_male_pose_0.obj')

    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", help="generate random STAR bodies")
    args = parser.parse_args()
    if args.generate:
        generate_bodies('male', int(args.generate))
        generate_bodies('female', int(args.generate))

    medidas, curves, positions = generate_measures(pose_0, pose_1, male_mesh)

    for idx, position in enumerate(positions):
        CurveUtils.save_obj(f'objs/{idx}.obj', position)

    # applier = CurveApplier(medidas, curves)
    # applier.generate_measures("female")
    # applier.generate_measures("male")