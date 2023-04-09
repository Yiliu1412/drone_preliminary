from airsim.types import ImageRequest, Quaternionr, Vector3r
import airsim
import time
import numpy as np
import os
import pprint
import cv2
from scipy.spatial.transform import Rotation as R 
import math

class circle_finder:
    def __init__(self, airsim_client) -> None:
        self.client = airsim_client
        self.cx = 320
        self.cy = 240
        self.fx = 268.5
        self.fy = 268.5

    def get_uav_position_rotation_in_wc(self):
        state = self.client.getMultirotorState()

        quaternionr = state.kinematics_estimated.orientation
        w = quaternionr.w_val
        x = quaternionr.x_val
        y = quaternionr.y_val
        z = quaternionr.z_val
        tmp = [x, y, z, w]
        r = R.from_quat(tmp)
        rotation_matrix = r.as_matrix()

        position = state.kinematics_estimated.position
        position_list = []
        position_list.append(position.x_val)
        position_list.append(position.y_val)
        position_list.append(position.z_val)
        
        return position_list, rotation_matrix

    def get_rgb_depthperspective_image(self):
        png_image = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float = False, compress = False),
                                        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float = True, compress = False)])

        pics_rgb = png_image[0]
        
        depthperspective = png_image[1]
        depthperspective = airsim.get_pfm_array(depthperspective)
        return pics_rgb, depthperspective

    def get_circle_x_y(self, rgb, depthperspective):
        depthperspective[depthperspective > 8] = 0
        depthperspective = depthperspective.astype(np.uint8)
        depthperspective = cv2.equalizeHist(depthperspective)
        # print(depthperspective)
        cv2.imshow('depth', depthperspective)
        cv2.waitKey(1000)

        count = 0
        circle = [0, 0, 0]
        # 霍夫变换圆检测
        circles = None
        while circles is None:
            circles = cv2.HoughCircles(depthperspective, cv2.HOUGH_GRADIENT, 1, 
                                        30, param1=None, param2=30, minRadius=30, maxRadius=300) # 注意图片分辨率大小与圆半径检测

        circles = list(circles)
        circles.sort(key = lambda x:x[0][2], reverse = True)
        circle += circles[0][0]
        count += 1
        if(count >= 10):
            circle = circle / count 

        # for circle in circles[0]:
        x = int(circle[0])
        y = int(circle[1])
        r = int(circle[2])
        img1d = np.frombuffer(rgb.image_data_uint8, dtype=np.uint8) 
        img_rgb = img1d.reshape(rgb.height, rgb.width, 3)
        rgb_pic = cv2.circle(img_rgb, (x, y), r, (0, 0, 255), 3) # 显示圆
        rgb_pic = cv2.circle(rgb_pic, (x, y), 2, (255, 255, 0), -1) # 显示圆心
        cv2.imshow('new', rgb_pic)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        return circle[0], circle[1] # 输出检测到的圆的x, y坐标

    def get_circle_center_z(self, depthperspective):
        shape = depthperspective.shape
        mask = cv2.inRange(depthperspective, 1, 8)
        tmp = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                if mask[i][j] != 0:
                    k1 = (j - self.cx) / self.fx
                    k2 = (i - self.cy) / self.fy
                    z = depthperspective[i][j] / math.sqrt(k1**2 + k2**2 + 1) 
                    tmp.append(z)
        circle_center_z = sum(tmp) / len(tmp)
        return circle_center_z

    def circle_cc_to_wc(self, pixel_x, pixel_y, z, t, R):
        camera_inner_matrix = [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
        camera_inner_matrix = np.linalg.pinv(np.array(camera_inner_matrix))
        print('pixels', pixel_x, pixel_y)
        point2D_h = [pixel_x, pixel_y, 1]
        point = (np.array(point2D_h) * z).T
        print("point2D:", point)
        tmp = np.dot(camera_inner_matrix, point)
        tmp[2] += 0.6
        print("circle in camera coord", tmp)
        self.R_b_c = [[0, 0 ,1], [1, 0, 0], [0, 1, 0]]
        tmp = np.dot(self.R_b_c, tmp)
        print("circle in body coord", tmp)
    
        result = np.dot(R, tmp) + np.array(t).T
        print("circle in world frame", list(result))
        return list(result)

    def get_circle_position_in_wc(self):
        position_list, rotation_matrix = self.get_uav_position_rotation_in_wc()
        pics_rgb, depthperspective = self.get_rgb_depthperspective_image()
        circle_xy = self.get_circle_x_y(pics_rgb, depthperspective)
        # print("circle_xy", circle_xy)

        circle_z = self.get_circle_center_z(depthperspective)
        
        result = self.circle_cc_to_wc(circle_xy[0], circle_xy[1], circle_z, position_list, rotation_matrix)

        return result


