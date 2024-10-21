import numpy as np
import os
import open3d as o3d
def get_random_rotations(num):
    """
    Get random rotations
    :return: random rotations
    """
    m = 13 * num
    x0 = np.random.rand(m, 1)
    x1 = np.random.rand(m, 1)
    x2 = np.random.rand(m, 1)
    theta1 = 2 * np.pi * x1
    theta2 = 2 * np.pi * x2

    s1 = np.sin(theta1)
    s2 = np.sin(theta2)
    c1 = np.cos(theta1)
    c2 = np.cos(theta2)

    r1 = np.sqrt(1-x0)
    r2 = np.sqrt(x0)

    # quats = [np.dot(np.array([r1[i], r2[i]*s1[i], r2[i]*c1[i], 0]), np.array([r1[i], r2[i]*s2[i], r2[i]*c2[i], 0])) for i in range(m)]
    quats = np.array([s1*r1, c1*r1, s2*r2, c2*r2])
    quats = np.resize(quats, (4,m)).transpose()
    return quats

def generate_random_translations(focal_length, sensor_size, target_size, roi_size, num):
    """
    Generate random translations
    :param focal_length: focal length of the camera
    :param sensor_size: sensor size of the camera
    :param roi_size: roi size of the target in camera coordinate system
    :return: random translations
    """
    m = 13 * num
    sensor_mean_size = np.mean(sensor_size)
    roi_mean_size = np.mean(roi_size) * sensor_mean_size

    z0 = target_size * focal_length / roi_mean_size
    z = np.random.normal(z0, 30, 1000)
    z_select = z[(z>30) & (z<35)]
    z_final = np.random.choice(z_select, m)

    z_array = np.zeros((len(z_final), 3, 1))
    z_array[:,2,:] = z_final.reshape((len(z_final), 1))

    alpha0 = np.arctan(sensor_size[0]/2 / focal_length)
    alpha = np.random.normal(0, alpha0, 1000)
    alpha_select = alpha[(alpha>-alpha0/2) & (alpha<alpha0/2)]
    alpha_final = np.random.choice(alpha_select, m)
    alpha_array = np.zeros((len(alpha_final), 3, 3))
    alpha_array[:,0,0] = np.cos(alpha_final)
    alpha_array[:,0,2] = -np.sin(alpha_final)
    alpha_array[:,1,1] = 1
    alpha_array[:,2,0] = np.sin(alpha_final)
    alpha_array[:,2,2] = np.cos(alpha_final)

    beta0 = np.arctan(sensor_size[1]/2 / focal_length)
    beta = np.random.normal(0, beta0, 1000)
    beta_select = beta[(beta>-beta0/2) & (beta<beta0/2)]
    beta_final = np.random.choice(beta_select, m)
    beta_array = np.zeros((len(beta_final), 3, 3))
    beta_array[:,0,0] = 1
    beta_array[:,1,1] = np.cos(beta_final)
    beta_array[:,1,2] = np.sin(beta_final)
    beta_array[:,2,1] = -np.sin(beta_final)
    beta_array[:,2,2] = np.cos(beta_final)

    trans = np.matmul(np.matmul(alpha_array, beta_array), z_array)
    trans = trans.squeeze(2)

    return trans

def quaternion_multiply(q1,q2):
    """
    Multiply two quaternions
    :param q1: quaternion 1
    :param q2: quaternion 2
    :return: result of multiplication
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quaternion_rotate_vector(q, v):
    """
    Rotate a vector by a quaternion
    :param q: quaternion
    :param v: vector
    :return: rotated vector
    """
    vec_quat = np.append(0,v)
    v_rotation = np.zeros((len(q), 3))
    q_inv = np.array([q[:,0], -q[:,1], -q[:,2], -q[:,3]]).transpose()
    for i in range(len(q)):
        v_rotation[i,:] = quaternion_multiply(quaternion_multiply(q[i], vec_quat), q_inv[i])[1:]

    # this is to visualize the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v_rotation)
    o3d.visualization.draw_geometries([pcd])




if __name__ == '__main__':
    num_of_each_category = 300
    quats = get_random_rotations(num_of_each_category)
    # this is to visualize the point cloud, which is the result of rotating the vector (1,0,0) by the quaternion
    # quaternion_rotate_vector(quats, np.array([1, 0, 0]))
    # 130 means the 13 classes and 10 samples for each class
    trans = generate_random_translations(0.01, np.array([0.007, 0.007]), 16, np.array([0.5, 0.5]), num_of_each_category)
    np.savetxt('pose_512.txt', np.hstack((quats, trans)), delimiter=',')
