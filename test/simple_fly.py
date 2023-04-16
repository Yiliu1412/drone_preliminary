import time
import numpy as np
import math
import cv2
import airsim
from scipy.spatial.transform import Rotation

circle_point = [
    (15.5, -19.6, -3.5, 5),
    (22, -41.2, -2.5, 5),
    (21, -61.5, -2, 5),
    (10, -78.2, -2, 5),
    (-9.3, -93, -2.5, 5),
    (-27, -98, -4, 5),
    (-50.1, -103, -5.7, 5)
]

circle_yaw = [
    -90,
    -90,
    -100,
    -120,
    -170,
    -170,
    -180
]

camera_param = {
    'cx': 320, 'cy': 240,
    'fx': 268.5, 'fy': 268.5
}

cx, cy, fx, fy = 320, 240, 268.5, 268.5

camera_inner_matrix = np.array([
    [268.5, 0, 320],
    [0, 268.5, 240],
    [0, 0, 1]
])

def get_image(client: airsim.MultirotorClient):
    scene_image, depth_image = client.simGetImages([
        airsim.ImageRequest('0', airsim.ImageType.Scene, pixels_as_float = False, compress = False),
        airsim.ImageRequest('0', airsim.ImageType.DepthPerspective, pixels_as_float = True, compress = False)
    ])

    scene_data = np.frombuffer(scene_image.image_data_uint8, dtype = np.uint8).reshape(scene_image.height, scene_image.width, 3)
    depth_data = np.asarray(depth_image.image_data_float, dtype = np.float32).reshape(depth_image.height, depth_image.width)

    return scene_data, depth_data

def find_circle_position_pnp(scene_image: np.ndarray, depth_image: np.ndarray):
    hsv = cv2.cvtColor(scene_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])) + cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))

    #cv2.imshow('color0', mask)
    #cv2.waitKey()

    circles = cv2.HoughCircles(
        mask, cv2.HOUGH_GRADIENT, 1, 30,
        param1 = None, param2 = 30,
        minRadius = 30, maxRadius = 0
    )

    x, y, r = circles[0][0]
    scene_image = cv2.circle(scene_image, (int(x), int(y)), int(r), (255, 0, 0), 3)
    scene_image = cv2.circle(scene_image, (int(x), int(y)), int(r + 10), (0, 255, 0), 3)
    scene_image = cv2.circle(scene_image, (int(x), int(y)), int(r - 10), (0, 0, 255), 3)
    scene_image = cv2.circle(scene_image, (int(x), int(y)), 2, (255, 255, 0), -1)

    cv2.imshow('color1', scene_image)
    cv2.waitKey()

    cv2.destroyAllWindows()
    
    
    '''zs = []
    h, w = scene_image.shape[0], scene_image.shape[1]
    for py in range(h):
        for px in range(w):
            if mask[py][px] != 0:
                kx = (px - cx) / fx
                ky = (py - cy) / fy
                z = depth_image[py][px] / math.sqrt(kx ** 2 + ky ** 2 + 1)
                zs.append(z)
    z = sum(zs) / len(zs)'''

    zs = []
    print(x, y, r)
    w, h = depth_image.shape
    step = 2 * math.pi / 360
    t = -step
    while t < 2 * math.pi:
        t += step
        dr = - r / 10
        min_z = 9999
        while dr < r / 10:
            dr += r / 100
            px = round(x + (r + dr) * math.cos(t))
            py = round(y + (r + dr) * math.sin(t))
            if px >= w or px < 0 or py >= h or py < 0:
                continue
            kx = (px - cx) / fx
            ky = (py - cy) / fy
            z = depth_image[py][px] / math.sqrt(kx ** 2 + ky ** 2 + 1)
            min_z = min(min_z, z)
        zs.append(min_z)
    z = sum(zs) / len(zs)
    print(x, y, z, r)

    return x, y, z, r

def find_circle_position_hough(scene_image: np.ndarray, depth_image: np.ndarray):
    #cv2.imshow('depth0', depth_data)
    depth_image[depth_image > 8] = 0
    depth_image = depth_image.astype(np.uint8)
    #cv2.imshow('depth1', depth_data)
    depth_data = cv2.equalizeHist(depth_image)
    cv2.imshow('depth2', depth_data)
    cv2.waitKey()

    circles = cv2.HoughCircles(
        depth_data, cv2.HOUGH_GRADIENT, 1, 30,
        param1 = None, param2 = 30,
        minRadius = 30, maxRadius = 300
    )

    #print(len(circles))
    #for x, y, r in circles[0]:
    #    print(x, y, r, ((x - 320) ** 2 + (y - 240) ** 2) ** 0.5)

    #max_x, max_y, max_r = circles[0][0]
    #for x, y, r in circles[0][1:]:
    #    if r > max_r:
    #        max_x, max_y, max_r = x, y, r

    scene_image = cv2.circle(scene_image, (int(circles[0][0][0]), int(circles[0][0][1])), int(circles[0][0][2]), (0, 0, 255), 3)
    scene_image = cv2.circle(scene_image, (int(circles[0][0][0]), int(circles[0][0][1])), 2, (255, 255, 0), -1)

    cv2.imshow('image0', scene_image)

    cv2.waitKey()
    cv2.destroyAllWindows()

    x, y, r = circles[0][0]
    return x, y, r

def get_world_coordinate(x: float, y: float, z: float, position: np.ndarray, rotation_matrix: np.ndarray):
    pinv = np.linalg.pinv(camera_inner_matrix)
    point = (np.array([x, y, 1]) * z).T
    tmp = np.dot(pinv, point)
    #tmp[2] += 0.6 #?
    rbc = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    result = np.dot(rotation_matrix, np.dot(rbc, tmp)) + position.T
    return result

def get_local_coordinate(x: float, y: float, z: float, position: np.ndarray, rotation_matrix: np.ndarray):
    pinv = np.linalg.pinv(camera_inner_matrix)
    point = (np.array([x, y, 1]) * z).T
    tmp = np.dot(pinv, point)
    # tmp[2] += 0.6 #?
    rbc = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    return np.dot(rbc, tmp)

def get_position_from_state(state: airsim.MultirotorState):
    return np.array([
        state.kinematics_estimated.position.x_val,
        state.kinematics_estimated.position.y_val,
        state.kinematics_estimated.position.z_val
    ])

def get_rotation_from_state(state: airsim.MultirotorState):
    return Rotation.from_quat([
        state.kinematics_estimated.orientation.x_val,
        state.kinematics_estimated.orientation.y_val,
        state.kinematics_estimated.orientation.z_val,
        state.kinematics_estimated.orientation.w_val
    ]).as_matrix()

def get_device_state(client: airsim.MultirotorClient):
    state = client.getMultirotorState()
    position = np.array([
        state.kinematics_estimated.position.x_val,
        state.kinematics_estimated.position.y_val,
        state.kinematics_estimated.position.z_val
    ])
    return position

def get_camera_info(client: airsim.MultirotorClient):
    info = client.simGetCameraInfo('0')
    position = [
        info.pose.position.x_val,
        info.pose.position.y_val,
        info.pose.position.z_val
    ]
    orientation = [
        info.pose.orientation.x_val,
        info.pose.orientation.y_val,
        info.pose.orientation.z_val,
        info.pose.orientation.w_val
    ]
    return np.array(position), Rotation.from_quat(orientation).as_matrix()

def normalize(v: np.ndarray):
    return v / math.sqrt(np.sum(v ** 2))

def calc_control_force(p_des, v_des, a_des, yaw_des, p_now, v_now, R_now, omega_now, m):
    kp = 2
    kv = 2
    kR = 0.4
    komega = 0.08
    e_p = p_now - p_des
    e_v = v_now - v_des
    g = 9.81
    e3 = np.array([[0], [0], [1]])
    # 求合力 f
    acc = -kp*e_p -kv*e_v - m*g*e3 + m*a_des   # 3x1
    f = -np.dot((acc).T, np.dot(R_now, e3))
    # 求期望的旋转矩阵 R_des
    proj_xb = np.array([math.cos(yaw_des), math.sin(yaw_des), 0])
    acc = acc.reshape(3)
    z_b = - acc / np.linalg.norm(acc)
    y_b = np.cross(z_b, proj_xb)
    y_b = y_b / np.linalg.norm(y_b)
    x_b = np.cross(y_b, z_b)
    x_b = x_b / np.linalg.norm(x_b)
    R_des = np.hstack([np.hstack([x_b.reshape([3, 1]), y_b.reshape([3, 1])]), z_b.reshape([3, 1])])
    # 求合力矩 M
    e_R_tem = np.dot(R_des.T, R_now) - np.dot(R_now.T, R_des)/2
    e_R = np.array([[e_R_tem[2, 1]], [e_R_tem[0, 2]], [e_R_tem[1, 0]]])
    M = -kR * e_R - komega * omega_now
    return f[0, 0], M

def get_pwm_control(f, M):
    mat = np.array([[4.179446268,       4.179446268,        4.179446268,        4.179446268],
                    [-0.6723341164784,  0.6723341164784,    0.6723341164784,    -0.6723341164784],
                    [0.6723341164784,   -0.6723341164784,   0.6723341164784,    -0.6723341164784],
                    [0.055562,          0.055562,           -0.055562,          -0.055562]])
    fM = np.vstack([f, M])
    u = np.dot(np.linalg.inv(mat), fM)
    u1 = u[0, 0]
    u2 = u[1, 0]
    u3 = u[2, 0]
    u4 = u[3, 0]
    return u1, u2, u3, u4

if __name__ == '__main__':
    client = airsim.MultirotorClient('127.0.0.1')
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    client.takeoffAsync().join()

    for i in range(len(circle_point)):
        print('=========================')
        print('CIRCLE {}'.format(i))

        client.moveToPositionAsync(*circle_point[i], yaw_mode = {
            'is_rate': False,
            'yaw_or_rate': 0.0
        }).join()
        #client.hoverAsync().join()
        time.sleep(3)
        client.rotateToYawAsync(circle_yaw[i]).join()
        #client.hoverAsync().join()
        # time.sleep(3)
        state = client.getMultirotorState()

        position, rotation = get_camera_info(client)

        print('x')
        scene_image, depth_image = get_image(client)

        cx, cy, cz, r = find_circle_position_pnp(scene_image, depth_image)

        wx, wy, wz = get_world_coordinate(cx, cy, cz, position, rotation)
        lx, ly, lz = get_local_coordinate(cx, cy, cz, position, rotation)

        current_position = get_device_state(client)
        v = normalize(np.array([wx, wy, wz]) - current_position) * 5
        lv = normalize(np.array([lx, ly, lz]))

        print(client.getMultirotorState().kinematics_estimated.position, v)
        
        client.simPlotPoints(
            [airsim.Vector3r(wx, wy, wz)],
            color_rgba=[1.0, 0.0, 1.0, 1.0],
            size=10.0,
            duration=30.0,
            is_persistent=True
        )

        client.moveToPositionAsync(wx, wy, wz, 5).join()
        #client.moveByVelocityAsync(v[0], v[1], v[2], 3).join()
        #client.moveByVelocityZAsync(v[0], v[1], wz, 3).join()
        #client.moveByRollPitchYawThrottleAsync(0.0, math.pi / 2 + math.atan(lz / lx), math.atan(ly / lx), 1.0, 3).join()
        client.hoverAsync().join()
        #time.sleep(5)

        print(client.getMultirotorState().kinematics_estimated.position, [wx, wy, wz])

        #find_circle_position_hough(scene_image, depth_image)
        #client.rotateToYawAsync(0).join()
        #time.sleep(3)

    client.armDisarm(False)
    client.enableApiControl(False)
