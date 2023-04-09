import time
import numpy as np
import math
import cv2
import airsim

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

def get_image(client: airsim.MultirotorClient):
    return client.simGetImages([
        airsim.ImageRequest('0', airsim.ImageType.Scene, pixels_as_float = False, compress = False),
        airsim.ImageRequest('0', airsim.ImageType.DepthPerspective, pixels_as_float = True, compress = False)
    ])

def find_circle_position_pnp(scene_image: airsim.ImageResponse, depth_image: airsim.ImageResponse):
    scene_data = np.frombuffer(scene_image.image_data_uint8, dtype = np.uint8).reshape(scene_image.height, scene_image.width, 3)

    hsv = cv2.cvtColor(scene_data, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])) + cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    
    cv2.imshow('color0', mask)
    cv2.waitKey()

def find_circle_position_hough(scene_image: airsim.ImageResponse, depth_image: airsim.ImageResponse):
    depth_data = airsim.get_pfm_array(depth_image)
    #cv2.imshow('depth0', depth_data)
    depth_data[depth_data > 8] = 0
    depth_data = depth_data.astype(np.uint8)
    #cv2.imshow('depth1', depth_data)
    depth_data = cv2.equalizeHist(depth_data)
    cv2.imshow('depth2', depth_data)
    cv2.waitKey()

    circles = cv2.HoughCircles(
        depth_data, cv2.HOUGH_GRADIENT, 1, 30,
        param1 = None, param2 = 30,
        minRadius = 30, maxRadius = 300
    )

    print(len(circles))
    for x, y, r in circles[0]:
        print(x, y, r, ((x - 320) ** 2 + (y - 240) ** 2) ** 0.5)
    print(depth_image.height, depth_image.width)
    print(scene_image.height, scene_image.width)

    scene_data = np.frombuffer(scene_image.image_data_uint8, dtype = np.uint8).reshape(scene_image.height, scene_image.width, 3)
    
    #max_x, max_y, max_r = circles[0][0]
    #for x, y, r in circles[0][1:]:
    #    if r > max_r:
    #        max_x, max_y, max_r = x, y, r
    
    scene_data = cv2.circle(scene_data, (int(circles[0][0][0]), int(circles[0][0][1])), int(circles[0][0][2]), (0, 0, 255), 3)
    scene_data = cv2.circle(scene_data, (int(circles[0][0][0]), int(circles[0][0][1])), 2, (255, 255, 0), -1)
    
    cv2.imshow('image0', scene_data)

    cv2.waitKey()
    cv2.destroyAllWindows()

    x, y, r = circles[0][0]
    return x, y, r

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
            'is_rate': True,
            'yaw_or_rate': 90.0
        }).join()
        # client.hoverAsync().join()
        # time.sleep(3)
        client.rotateToYawAsync(circle_yaw[i]).join()
        client.hoverAsync().join()
        # time.sleep(3)
        state = client.getMultirotorState()
        print(state)
        scene_image, depth_image = get_image(client)
        find_circle_position_pnp(scene_image, depth_image)
        find_circle_position_hough(scene_image, depth_image)
        client.rotateToYawAsync(0).join()
        time.sleep(3)

    client.armDisarm(False)
    client.enableApiControl(False)
