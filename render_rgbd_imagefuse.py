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

import bpy
import mathutils
import numpy as np
import os
import sys
import time


def random_pose():
    angle_x = np.random.uniform() * 2 * np.pi
    angle_y = np.random.uniform() * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(R[:, 2], 1)
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    return pose

def quaternion2roatation(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    R = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                  [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                  [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])
    return R

def pose_to_transformation_matrix(input_data):
    """
    Convert pose to transformation matrix
    :param position: 3D position
    :param quaternion: quaternion
    :return: transformation matrix
    """
    quaternion = input_data[0:4]
    translation = input_data[4:7]
    transformation_matrix = np.zeros((4, 4))
    transformation_matrix[:3, :3] = quaternion2roatation(quaternion)
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1
    return transformation_matrix

def setup_blender(width, height, focal_length):
    # camera
    camera = bpy.data.objects['Camera']
    camera.data.clip_end = 1000
    # camera.data.angle = np.arctan(width / 2 / focal_length) * 2
    camera.data.lens = focal_length
    # camera.rotation_mode = 'QUATERNION'
    camera.data.sensor_fit = 'AUTO'
    camera.data.sensor_width = 7

    # render layer
    scene = bpy.context.scene
    scene.render.filepath = 'buffer'
    scene.view_layers['ViewLayer'].use_pass_z = True
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.use_zbuffer = True
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = width
    scene.render.resolution_y = height

    # this is for transparent background
    scene.render.film_transparent = True

    sun_data = bpy.data.lights.new(name='Sun1', type='SUN')
    sun_data.energy = 0.3
    Sun_object = bpy.data.objects.new(name='Sun1', object_data=sun_data)
    Sun_object.rotation_mode = 'QUATERNION'
    Sun_object.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(Sun_object)


    # # compositor nodes
    # scene.use_nodes = True
    # tree = scene.node_tree
    # # remove nodes before create new nodes
    # for n in tree.nodes:
    #     tree.nodes.remove(n)
    #
    # # create new nodes
    # render_layers = tree.nodes.new('CompositorNodeRLayers')
    #
    # depth_file_output = tree.nodes.new('CompositorNodeOutputFile')
    # depth_file_output.label = 'Depth Output'
    # depth_file_output.format.file_format = 'OPEN_EXR'
    # depth_file_output.format.color_depth = '32'
    # depth_file_output.format.use_zbuffer = True
    # depth_file_output.base_path = ''
    # tree.links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    #
    # image_file_output = tree.nodes.new('CompositorNodeOutputFile')
    # image_file_output.label = 'Image'
    # image_file_output.base_path = ''
    # tree.links.new(render_layers.outputs['Image'], image_file_output.inputs[0])

    # remove default cube
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()
    # remove default light
    bpy.data.objects['Light'].select_set(True)
    bpy.ops.object.delete()

    bpy.context.scene.render.engine = 'CYCLES'
    # bpy.context.scene.cycles.feature_set = 'EXPERIMENTAL'
    bpy.context.scene.cycles.device = 'CPU'

    # return scene, camera, Sun_object, depth_file_output, image_file_output
    return scene, camera, Sun_object


def generate_composeitor_nodes(img_file):
    scene.use_nodes = True
    tree = scene.node_tree
    # remove nodes before create new nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create new nodes
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    image_node = tree.nodes.new('CompositorNodeImage')
    image_node.image = bpy.data.images.load(img_file)

    # create alpha over node
    alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')
    scale_node = tree.nodes.new('CompositorNodeScale')

    scale_node.space = 'RENDER_SIZE'
    scale_node.frame_method = 'CROP'

    # connect nodes
    tree.links.new(image_node.outputs['Image'], scale_node.inputs[0])
    tree.links.new(scale_node.outputs['Image'], alpha_over_node.inputs[1])
    tree.links.new(render_layers.outputs['Image'], alpha_over_node.inputs[2])


    depth_file_output = tree.nodes.new('CompositorNodeOutputFile')
    depth_file_output.label = 'Depth Output'
    depth_file_output.format.file_format = 'OPEN_EXR'
    depth_file_output.format.color_depth = '32'
    depth_file_output.format.use_zbuffer = True
    depth_file_output.base_path = ''
    tree.links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

    image_file_output = tree.nodes.new('CompositorNodeOutputFile')
    image_file_output.label = 'Image'
    image_file_output.base_path = ''
    tree.links.new(alpha_over_node.outputs['Image'], image_file_output.inputs[0])

    return depth_file_output, image_file_output

if __name__ == '__main__':
    # model_dir = sys.argv[-4]
    # list_path = sys.argv[-3]
    # output_dir = sys.argv[-2]
    # num_scans = int(sys.argv[-1])

    model_dir = './available_model/'
    output_dir = './scan_data_sim_satellite_512/'
    pose_dir = './pose_512.txt'
    list_path = './model_list.txt'
    image_file = './background/truth/'
    num_scans = 10

    nn = 1
    width = 512 * nn
    height = 512 * nn
    focal = 10 * nn
    scene, camera, Sun = setup_blender(width, height, focal)
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])

    with open(os.path.join(list_path)) as file:
        model_list = [line.strip() for line in file]

    pose_data = []
    with open(os.path.join(pose_dir)) as file:
        for line in file:
            parts = line.strip().split(',')
            txt_data = [float(p) for p in parts]
            pose_data.append(txt_data)

    img_file_name = os.listdir(image_file)

    pose_data = np.array(pose_data)
    quaternions = pose_data[:, 0:4]
    translations = pose_data[:, 4:7]
    # print(pose_data[0,:])

    open('blender.log', 'w+').close()
    # os.system('rm -rf %s' % output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(os.path.join(output_dir, 'intrinsics.txt'), intrinsics, '%f')

    j=0
    for model_id in model_list:
        start = time.time()
        exr_dir = os.path.join(output_dir, model_id, 'exr')
        rgb_dir = os.path.join(output_dir, model_id, 'rgb')
        pose_dir = os.path.join(output_dir, model_id, 'pose')
        os.makedirs(exr_dir)
        os.makedirs(pose_dir)
        os.makedirs(rgb_dir)

        # Redirect output to log file
        old_os_out = os.dup(1)
        os.close(1)
        os.open('blender.log', os.O_WRONLY)

        # Import mesh model
        model_path = os.path.join(model_dir, model_id+'.obj')
        bpy.ops.import_scene.obj(filepath=model_path)
        obj = bpy.context.selected_objects[0]
        # obj.name = 'Model'
        # obj.data.name = 'Model'
        obj.rotation_mode = 'QUATERNION'
        #

        # Rotate model by 90 degrees around x-axis (z-up => y-up) to match ShapeNet's coordinates
        # bpy.ops.transform.rotate(value=-np.pi / 2, orient_axis='X')

        # Render

        for i in range(num_scans):
            scene.frame_set(i)
            # pose = pose_to_transformation_matrix(pose_data[i,:])
            # print(quaternions[i,:])
            Sun.rotation_quaternion = quaternions[j,:]

            img_file = os.path.join(image_file, img_file_name[i])
            depth_file_output, image_file_output = generate_composeitor_nodes(img_file)

            # camera.matrix_world = mathutils.Matrix(pose)
            obj.rotation_quaternion = quaternions[j,:]
            obj.location = translations[j,:]
            camera.rotation_euler = (np.pi,0,0)
            camera.location = (0,0,0)
            j += 1
            # pose_data = pose_data[i, 0:8]
            # scene.render.filepath = exr_dir
            depth_file_output.file_slots[0].path = os.path.join(exr_dir, '#.exr')
            image_file_output.file_slots[0].path = os.path.join(rgb_dir, '#.png')
            # depth_file_output.file_slots[0].path = os.path.join(exr_dir, '#.exr')
            bpy.ops.render.render(write_still=True)
            # np.savetxt(os.path.join(pose_dir, '%d.txt' % i), pose, '%f')

        # Clean up
        bpy.ops.object.delete()
        for m in bpy.data.meshes:
            bpy.data.meshes.remove(m)
        for m in bpy.data.materials:
            m.user_clear()
            bpy.data.materials.remove(m)

        # Show time
        os.close(1)
        os.dup(old_os_out)
        os.close(old_os_out)
        print('%s done, time=%.4f sec' % (model_id, time.time() - start))