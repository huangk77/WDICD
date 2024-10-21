'''
MIT License

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import Imath
import OpenEXR
import argparse
import array
import numpy as np
import os
import open3d as o3d
import cv2
from tqdm import tqdm
from PIL import Image
import imgviz
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R

def read_exr(exr_path, height, width):
    File = OpenEXR.InputFile(exr_path)
    # PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    # DW = File.header()['dataWindow']
    # Size = (height, width)
    # rgb = [np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    # r = np.reshape(rgb[0],(height, width))
    # mytiff = np.zeros((height, width), dtype=np.float32)
    # mytiff = r
    depth_arr = array.array('f', File.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    # depth_arr = array.array('f', File.channel('R', Imath.PixelType(Imath.PixelType.DOUBLE)))
    depth = np.array(depth_arr).reshape((height, width))
    depth[depth < 0] = 0
    depth[np.isinf(depth)] = 0
    depth[depth > 1000] = 0
    return depth


def depth2pcd(depth, intrinsics):
    # depth = np.flipud(depth)
    y, x = np.where((depth > 0) & (depth < 1000))
    camera_cx = intrinsics[0, 2]
    camera_cy = intrinsics[1, 2]
    camera_fx = intrinsics[0, 0]
    camera_fy = intrinsics[1, 1]
    # sensor size / pixel number
    dx = 36 / (2 * camera_cx)
    # dx = 7 / (2 * camera_cx)
    # dy = 24 / (2 * camera_cy)
    dy = dx
    points_z = depth[y, x]
    points_x = points_z * (x - camera_cx) * dx / camera_fx
    points_y = points_z * (y - camera_cy) * dy / camera_fy
    points = np.stack([points_x, points_y, points_z], axis=0)

    return points

def euler_to_rotation_matrix(euler):
    """
    Convert euler to rotation matrix
    :param euler: euler
    :return: rotation matrix
    """
    r = R.from_euler('xyz', euler, degrees=False)
    rotation_matrix = r.as_matrix()

    return rotation_matrix

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert quaternion to rotation matrix
    :param quaternion: quaternion
    :return: rotation matrix
    """
    q = quaternion
    rotation_matrix = np.array([[1-2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[3]*q[0]), 2*(q[1]*q[3]+q[2]*q[0])],
                                [2*(q[1]*q[2]+q[3]*q[0]), 1-2*(q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[1]*q[0])],
                                [2*(q[1]*q[3]-q[2]*q[0]), 2*(q[2]*q[3]+q[1]*q[0]), 1-2*(q[1]**2+q[2]**2)]])
    return rotation_matrix

def depth2mask(model_dir, pose_dir, quaternion, translation, R_img, R_cam, depth, intrinsics):

    model_data = np.loadtxt(model_dir, delimiter=',')
    model_points = model_data[:, :3]
    model_labels = model_data[:, 3]

    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    model_points_rotation = np.dot(rotation_matrix, model_points.T).T + translation

    # depth = np.flipud(depth)
    height, width = depth.shape
    y, x = np.where((depth > 0) & (depth < 1000))
    camera_cx = intrinsics[0, 2]
    camera_cy = intrinsics[1, 2]
    camera_fx = intrinsics[0, 0]
    camera_fy = intrinsics[1, 1]
    # sensor size / pixel number
    # dx = 36 / (2 * camera_cx)
    dx = 7 / (2 * camera_cx)
    # dy = 24 / (2 * camera_cy)
    dy = dx
    points_z = depth[y, x]
    points_x = points_z * (x - camera_cx) * dx / camera_fx
    points_y = points_z * (y - camera_cy) * dy / camera_fy
    points = np.stack([points_x, points_y, points_z], axis=0)

    mask = np.zeros((height, width))
    if points.size >0 :
        scan_data = points.T
        scan_data_img = np.dot(R_img, scan_data.T).T
        scan_points_camera = np.dot(R_cam, scan_data_img.T).T

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(model_points_rotation)
        _, idx = nbrs.kneighbors(scan_points_camera)

        num = points.shape[1]
        scan_points_label = np.zeros(num)
        scan_points_label = model_labels[idx]
        mask[y, x] = scan_points_label.squeeze(1)

    return mask, points


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('list_file', default='')
    parser.add_argument('--intrinsics_file', default="./scan_data_sim_satellite_512/intrinsics.txt")
    parser.add_argument('--output_dir', default="./scan_data_sim_satellite_512/")
    parser.add_argument('--model_dir', default="./available_model/label_data/")
    parser.add_argument('--pose_dir', default="./pose_512.txt")
    parser.add_argument('--list_file', default="./model_list.txt")
    parser.add_argument('--num_scans', type=int, default=300)
    args = parser.parse_args()

    with open(args.list_file) as file:
        model_list = file.read().splitlines()
    # model_list = ['11', '23', '44']
    # model_list = model_list[1:]
    intrinsics = np.loadtxt(args.intrinsics_file)
    width = int(intrinsics[0, 2] * 2)
    height = int(intrinsics[1, 2] * 2)

    pose_dir = args.pose_dir
    pose_data = []
    with open(os.path.join(pose_dir)) as file:
        for line in file:
            parts = line.strip().split(',')
            txt_data = [float(p) for p in parts]
            pose_data.append(txt_data)

    pose_data = np.array(pose_data)
    quaternions = pose_data[:, :4]
    translations = pose_data[:, 4:]

    euler_img = [np.pi, 0, 0]
    R_img = euler_to_rotation_matrix(euler_img)
    euler_cam = [np.pi, 0, 0]
    R_cam = euler_to_rotation_matrix(euler_cam)
    j = 0
    for model_id in model_list:
        depth_dir = os.path.join(args.output_dir, model_id, 'depth')
        pcd_dir = os.path.join(args.output_dir, model_id, 'pcd')
        mask_dir = os.path.join(args.output_dir, model_id, 'mask')
        model_dir = os.path.join(args.model_dir, model_id + '.csv')
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(pcd_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        print("Processing model %s" % model_id)
        for i in tqdm(range(args.num_scans)):
            exr_path = os.path.join(args.output_dir, model_id, 'exr', '%d.exr' % i)
            # pose_path = os.path.join(args.output_dir, model_id, 'pose', '%d.txt' % i)
            quaternion = quaternions[j]
            translation = translations[j]

            depth = read_exr(exr_path, height, width)
            depth_uint32 = depth * 1000
            depth_uint32[depth_uint32 > (255*255*255*255 - 1)] = 255 * 255 * 255*255 - 1
            # cv2.imwrite(os.path.join(depth_dir, '%d.tiff' % i), depth_uint32.astype(np.float32))  # if depth > 65535, then uint16 return small data
            # depth_read = cv2.imread(os.path.join(depth_dir, '%d.tiff' % i), cv2.IMREAD_ANYDEPTH)
            # print('output:', depth_read.max())
            # cv2.imshow('depth',depth_read)
            # cv2.waitKey(0)
            # cv2.destroyWindow()

            # depth_img = o3d.geometry.Image(np.uint16(depth * 1000))
            # o3d.io.write_image(os.path.join(depth_dir, '%d.png' % i), depth_img)

            # pose = np.loadtxt(pose_path)
            mask, points = depth2mask(model_dir, pose_dir, quaternion, translation, R_img, R_cam, depth, intrinsics)
            # points = depth2pcd(depth, intrinsics)
            if points.size > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.T)
                # IMPORTANT: if the point cloud is perform like our scan data, change the save format of output depth from '16' to '32
                # o3d.visualization.draw_geometries([pcd])
                o3d.io.write_point_cloud(os.path.join(pcd_dir, '%d.pcd' % i), pcd)
            label = Image.fromarray(mask.astype(np.uint8), mode='P')  # 转换为PIL的P模式
            # 转换成VOC格式的P模式图像
            colormap = imgviz.label_colormap()
            label.putpalette(colormap.flatten())
            label.save(os.path.join(mask_dir, '%d.png' % i))
            j += 1

