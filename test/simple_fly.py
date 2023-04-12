import time
import numpy as np
import math
import cv2
import airsim
from scipy.spatial.transform import Rotation

circle_point = [
    (15.5, -19.6, -3.5, 5),
    (22, -41.2, -2.5, 5),
    (21, -61.5, -2, 3),
    (10, -78.2, -2, 3),
    (-9.3, -93, -2.5, 3),
    (-27, -98, -4, 3),
    (-50.1, -103, -5.7, 3)
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

    cv2.imshow('color0', mask)
    cv2.waitKey()

    circles = cv2.HoughCircles(
        mask, cv2.HOUGH_GRADIENT, 1, 30,
        param1 = None, param2 = 30,
        minRadius = 30, maxRadius = 300
    )

    x, y, r = circles[0][0]
    scene_image = cv2.circle(scene_image, (int(x), int(y)), int(r), (255, 0, 0), 3)
    scene_image = cv2.circle(scene_image, (int(x), int(y)), 2, (255, 255, 0), -1)

    cv2.imshow('color1', scene_image)
    cv2.waitKey()

    cv2.destroyAllWindows()

    t = 0
    sum, cnt = 0, 0
    print(x, y, r)
    while t < 2 * math.pi:
        px = int(x + r * math.cos(t))
        py = int(y + r * math.sin(t))
        kx = (px - cx) / fx
        ky = (py - cy) / fy
        z = depth_image[px][py] / math.sqrt(kx ** 2 + ky ** 2 + 1)
        sum += z
        cnt += 1
        t += 1
    temp_z = sum / cnt

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

def get_world_coordinate(x: float, y: float, z: float, position: list, rotation_matrix: np.ndarray):
    pinv = np.linalg.pinv(camera_inner_matrix)
    point = (np.array([x, y, 1]) * z).T
    tmp = np.dot(pinv, point)
    # tmp[2] += 0.6 #?
    rbc = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    result = np.dot(rotation_matrix, np.dot(rbc, tmp)) + np.array(position).T
    return result

def get_position_from_state(state: airsim.MultirotorState):
    return [
        state.kinematics_estimated.position.x_val,
        state.kinematics_estimated.position.y_val,
        state.kinematics_estimated.position.z_val
    ]

def get_rotation_from_state(state: airsim.MultirotorState):
    return Rotation.from_quat([
        state.kinematics_estimated.orientation.x_val,
        state.kinematics_estimated.orientation.y_val,
        state.kinematics_estimated.orientation.z_val,
        state.kinematics_estimated.orientation.w_val
    ]).as_matrix()

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
            #'is_rate': True,
            #'yaw_or_rate': 90.0
        }).join()
        #client.hoverAsync().join()
        time.sleep(6)
        client.rotateToYawAsync(circle_yaw[i]).join()
        #client.hoverAsync().join()
        # time.sleep(3)
        state = client.getMultirotorState()
        print(state)

        scene_image, depth_image = get_image(client)

        cx, cy, cz, r = find_circle_position_pnp(scene_image, depth_image)

        wx, wy, wz = get_world_coordinate(
            cx, cy, cz,
            get_position_from_state(state),
            get_rotation_from_state(state)
        )

        #client.moveToPositionAsync(wx, wy, wz, 1).join()
        #client.hoverAsync().join()
        #time.sleep(5)

        #find_circle_position_hough(scene_image, depth_image)
        #client.rotateToYawAsync(0).join()
        #time.sleep(3)

    client.armDisarm(False)
    client.enableApiControl(False)
