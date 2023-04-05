from airsim.types import ImageRequest, Quaternionr, Vector3r
import airsim
import time
import numpy as np
import os
import pprint
import cv2
from scipy.spatial.transform import Rotation as R 
import math
import circle_finder

class uav_setpoints:
    def __init__(self) -> None:
        self.circle_setpoint_moveToPositionAsync = (
            (15.5, -19.6 , -3.5, 5),
            (22, -41.2 , -2.5, 5),
            (21, -61.5 , -2, 3),
            (10, -78.2, -2, 3),
            (-9.3, -93, -2.5, 3),
            (-27, -98, -4, 3),
            (-50.1, -103, -5.7, 3)
        )

        self.circle_yaw_rotateToYawAsync = (
            -90,
            -90,
            -100,
            -120,
            -170,
            -170,
            -180
        )

        self.land_setpoint_moveToPositionAsync = (-62.9, -102.3, -4, 3)

    def get_circle_setpoint(self, id_from_one):
        return self.circle_setpoint_moveToPositionAsync[id_from_one - 1]

    def get_circle_yaw(self, id_from_one):
        return self.circle_yaw_rotateToYawAsync[id_from_one - 1]

    def get_land_setpoint(self):
        return self.land_setpoint_moveToPositionAsync

# ===========================

class airsim_client:
    def __init__(self, ip_addr='127.0.0.1') -> None:
        print("Try to connect {}...".format(ip_addr))
        self.client = airsim.MultirotorClient(ip_addr)
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        # self.dhc = dh.drone_func_class()
        self.circle_finder = circle_finder.circle_finder(self.client)

        self.setpoints = uav_setpoints()

    def task_takeoff(self):
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

    def task_cross_circle(self, circle_id_from_one):
        self.client.moveToPositionAsync(*self.setpoints.get_circle_setpoint(circle_id_from_one)).join()
        self.client.hoverAsync().join() # 悬停函数
        time.sleep(3) #
        # airsim.wait_key('Press any key to rotate')
        self.client.rotateToYawAsync(self.setpoints.get_circle_yaw(circle_id_from_one)).join() # 旋转yaw角，正对障碍物识别
        # time.sleep(3) #
        circle_xyz = self.circle_finder.get_circle_position_in_wc()
        self.client.moveToPositionAsync(*circle_xyz, 1).join()
        self.client.hoverAsync().join()
        # self.dhc.cross_circle(self.client)

        self.client.rotateToYawAsync(0).join()
        # time.sleep(3)
    
    def task_land(self):
        self.client.moveToPositionAsync(*self.setpoints.get_land_setpoint()).join()
        self.client.hoverAsync().join() # 悬停函数
        time.sleep(3) #
        self.client.landAsync().join()
        self.client.armDisarm(False)

    def begin_task(self):
        print("=========================")
        print("Taking off...")

        self.task_takeoff()
        for circle_id in range(1, 7 + 1):
            # time.sleep(3)
            print("=========================")
            print("Now try to pass circle {}...".format(circle_id))
            self.task_cross_circle(circle_id)
        
        print("=========================")
        print("Landing...")
        self.task_land()

        print("=========================")
        print("Task is finished.")

