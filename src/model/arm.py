from re import L
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
import pyrep
from pyrep.backend import sim, utils
from pyrep.objects import Object
from pyrep.objects.dummy import Dummy
from pyrep.robots.configuration_paths.arm_configuration_path import (
    ArmConfigurationPath)
from pyrep.robots.robot_component import RobotComponent
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.errors import ConfigurationError, ConfigurationPathError, IKError
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.const import PYREP_SCRIPT_TYPE
from typing import List, Union
import numpy as np
import warnings
import glob
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import math

from pyrep.objects.vision_sensor import VisionSensor
from scipy.spatial.transform import Rotation as R

# Hold simulation for Panda Arm
class SimPanda:
    # Scene: Location of scene file
    # Bottom: Bound the z coordinate when moving
    def __init__(self, scene, bottom, headless=False, vision_cam="wrist_cam") -> None:
        self.pr = PyRep()
        self.pr.launch(scene, headless=headless)
        self.pr.start()

        self.arm = Panda()
        self.gripper = PandaGripper()

        self.home_pos = Shape("home").get_position()
        euler_org = np.array([180., 0.,180.])
        r = R.from_euler("ZYX", euler_org, degrees = True)
        self.quat_org = r.as_quat()
        self.bottom = bottom

        self.cam = VisionSensor(vision_cam)
        
        self.pr.step()

        self.go_home()
        self.gripper_org = self.gripper.get_orientation()
        self.default_joint = self.arm.get_joint_positions()


    # Get image and depth from camera
    def get_im_d(self):
        im = self.cam.capture_rgb()
        d = self.cam.capture_depth()

        im*=255
        im = np.array(im, dtype = np.uint8)

        d*=5000
        d = np.array(d, dtype = np.uint8)

        return im, d

    # Reset to orginal joint positions
    def reset(self):
        self.arm.set_joint_positions(self.default_joint)
        self.reset_gripper()

    # Go to home position
    def go_home(self):
        self.move(self.home_pos)
        self.straight_ee()


    def straight_ee(self):
        pos, _ = self.arm.get_tip().get_position(), self.arm.get_tip().get_quaternion()
        new_joint_angles = self.arm.solve_ik_via_jacobian(pos, quaternion = self.quat_org)
        self.arm.set_joint_target_positions(new_joint_angles)
        self.pr.step()


    # Move to target (3d numpy array [X, Y, Z]) using path, if no quat is supplied use default
    def move(self, target, quat=None):
        target[2] = max(target[2], self.bottom)

        if quat is None:
            quat = self.quat_org
        try:
            path = self.arm.get_path(
                position=target, quaternion=quat)
        except ConfigurationPathError as e:
            print('Could not find path')
            return False

        # Step the simulation and advance the agent along the path
        done = False
        while not done:
            done = path.step()
            self.pr.step()

        return True


    # Move by delta (3d numpy array [X, Y, Z]) using set joint target positions, if no quat is supplied use default
    def move_by(self, delta, quat=None):
        if quat is None:
            quat = self.quat_org
        pos = self.arm.get_tip().get_position()
        pos += delta
        pos[2] = max(pos[2], self.bottom)
        new_joint_angles = self.arm.solve_ik_via_jacobian(pos, quaternion = quat)
        self.arm.set_joint_target_positions(new_joint_angles)
        self.pr.step()


    # Move delta (3d numpy array [X, Y, Z]) by using path, if no quat is supplied use default
    def move_by_smooth(self, delta):
        current_pos = self.get_position()
        return self.move(current_pos + delta)


    # Move down by 'by' (3d numpy array [X, Y, Z]) and try to pick up any object in objects
    def down_grasp(self, by, objects):
        """Go straight down and grasp"""
        times = int(by / 0.005)
        for i in range(times):
            self.move_by(np.array([0.,0.,-0.005]), self.arm.get_tip().get_quaternion())
        
        for object in objects:
            self.gripper.grasp(object)

        done = False
        while not done:
            done = self.gripper.actuate(0.0, velocity=0.2)
            self.pr.step()

        done = False

        for i in range(10):
            self.move_by(np.array([0.,0.,0.01]), self.arm.get_tip().get_quaternion())


    # Open gripper
    def open_gripper(self):
        done = False
        self.gripper.release()
        while not done:
            done = self.gripper.actuate(1, velocity=0.04)
            self.pr.step()

        for i in range(10):
            self.move_by(np.array([0.,0.,0.01]), self.arm.get_tip().get_quaternion())


    # Rotate gripper by delta (3d numpy array [X, Y, Z])
    def rotate_gripper_by(self, delta):
        self.gripper.rotate(delta)
        self.pr.step()


    # Set gripper to a rotation angle (3d numpy array [X, Y, Z])
    def set_rotation(self, angle):
        self.gripper.set_orientation(angle)
        self.pr.step()

    
    # Reset gripper to original
    def reset_gripper(self):
        self.set_rotation(self.gripper_org)
        self.pr.step()

    
    # Get position of tip
    def get_position(self):
        return self.arm.get_tip().get_position()


    # Get orientation of gripper
    def get_gripper_orientation(self):
        return self.gripper.get_orientation()

    
    # Get objects gripped
    def get_gripped(self):
        return self.gripper.get_grasped_objects()


    # Shutdown simulation
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()