#!/usr/bin/env python
import rospy
import numpy as np
import math
from sensor_msgs.msg import Imu

class KalmanFilterNode:
    def __init__(self):
        rospy.init_node('jetbot_kalman_filter', anonymous=True)

        self.imu_topic = rospy.get_param('~input_topic', '/imu')
        self.pub_topic = rospy.get_param('~output_topic', '/kalman_imu')
        
        # Tuning Defaults
        self.default_gyro_var = rospy.get_param('~default_gyro_var', 0.005) 
        self.default_accel_var = rospy.get_param('~default_accel_var', 0.05)
        
        # State Initialization
        self.x_est = np.array([[1., 0., 0., 0.]]).T 
        self.P_est = np.eye(4) * 0.1 
        self.C = np.eye(4) 
        self.last_time = None

        self.pub = rospy.Publisher(self.pub_topic, Imu, queue_size=10)
        self.sub = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
        
        rospy.loginfo("Kalman Filter running")

    def imu_callback(self, msg):
        # 1. Time Check
        current_time = msg.header.stamp.to_sec()
        if self.last_time is None:
            self.last_time = current_time
            return
        dt = current_time - self.last_time
        self.last_time = current_time
        if dt <= 0: return

        # 2. Raw Data
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        gx, gy, gz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z

        # 3. Handle Covariance (The Fix)
        cov_g = msg.angular_velocity_covariance
        avg_g = (cov_g[0] + cov_g[4] + cov_g[8]) / 3.0
        
        # If garbage, overwrite with default
        if avg_g > 1.0 or avg_g == 0.0: 
            gyro_var = self.default_gyro_var
        else:
            gyro_var = avg_g

        cov_a = msg.linear_acceleration_covariance
        avg_a = (cov_a[0] + cov_a[4] + cov_a[8]) / 3.0
        
        # If garbage, overwrite with default
        if avg_a > 1.0 or avg_a == 0.0: 
            accel_var = self.default_accel_var
        else:
            accel_var = avg_a

        # 4. Math Scales
        Q_scale = gyro_var * (dt / 2.0)**2
        R_scale = accel_var / (9.81**2)

        process_noise = np.eye(4) * Q_scale
        measure_noise = np.eye(4) * R_scale

        # 5. PREDICTION (Gyro)
        Un = np.array([
            [0, -gx, -gy, -gz],
            [gx,  0,  gz, -gy],
            [gy, -gz,  0,  gx],
            [gz,  gy, -gx,  0]
        ])
        A = np.eye(4) + (0.5 * dt * Un)
        
        x_pred = np.dot(A, self.x_est)
        P_pred = np.dot(np.dot(A, self.P_est), A.T) + process_noise
        x_pred = x_pred / np.linalg.norm(x_pred)

        # 6. MEASUREMENT (Accel)
        meas_roll = math.atan2(ay, az)
        meas_pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az) + 0.00001)
        
        # Manual Yaw Extraction
        qw, qx, qy, qz = x_pred[0,0], x_pred[1,0], x_pred[2,0], x_pred[3,0]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        pred_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Reconstruct Quaternion
        cy = math.cos(pred_yaw * 0.5)
        sy = math.sin(pred_yaw * 0.5)
        cp = math.cos(meas_pitch * 0.5)
        sp = math.sin(meas_pitch * 0.5)
        cr = math.cos(meas_roll * 0.5)
        sr = math.sin(meas_roll * 0.5)

        q_w = cr * cp * cy + sr * sp * sy
        q_x = sr * cp * cy - cr * sp * sy
        q_y = cr * sp * cy + sr * cp * sy
        q_z = cr * cp * sy - sr * sp * cy
        
        measured_quat = np.array([[q_w, q_x, q_y, q_z]]).T

        if np.dot(x_pred.T, measured_quat) < 0:
            measured_quat = -measured_quat

        # 7. UPDATE
        S = np.dot(np.dot(self.C, P_pred), self.C.T) + measure_noise
        K = np.dot(np.dot(P_pred, self.C.T), np.linalg.inv(S))

        y_res = measured_quat - np.dot(self.C, x_pred)
        x_new = x_pred + np.dot(K, y_res)
        P_new = np.dot((np.eye(4) - np.dot(K, self.C)), P_pred)

        x_new = x_new / np.linalg.norm(x_new)
        self.x_est = x_new
        self.P_est = P_new

        # 8. Publish (Passing the FIXED variances now)
        self.publish_filtered_msg(msg, x_new, P_new, gyro_var, accel_var)

    def publish_filtered_msg(self, original_msg, x_state, P_state, gyro_var, accel_var):
        filtered_msg = Imu()
        filtered_msg.header = original_msg.header
        
        # Orientation
        filtered_msg.orientation.w = x_state[0, 0]
        filtered_msg.orientation.x = x_state[1, 0]
        filtered_msg.orientation.y = x_state[2, 0]
        filtered_msg.orientation.z = x_state[3, 0]
        
        # Pass raw data
        filtered_msg.angular_velocity = original_msg.angular_velocity
        filtered_msg.linear_acceleration = original_msg.linear_acceleration
        
        # --- THE FIX ---
        # Instead of copying the garbage, we build new covariance matrices
        # with the "fixed" variances we calculated.
        
        # 1. Gyro Covariance (diagonal)
        filtered_msg.angular_velocity_covariance = [0.0] * 9
        filtered_msg.angular_velocity_covariance[0] = gyro_var
        filtered_msg.angular_velocity_covariance[4] = gyro_var
        filtered_msg.angular_velocity_covariance[8] = gyro_var
        
        # 2. Accel Covariance (diagonal)
        filtered_msg.linear_acceleration_covariance = [0.0] * 9
        filtered_msg.linear_acceleration_covariance[0] = accel_var
        filtered_msg.linear_acceleration_covariance[4] = accel_var
        filtered_msg.linear_acceleration_covariance[8] = accel_var

        # 3. Orientation Covariance (from Kalman P matrix)
        filtered_msg.orientation_covariance = [0.0] * 9
        filtered_msg.orientation_covariance[0] = P_state[1,1] # Roll
        filtered_msg.orientation_covariance[4] = P_state[2,2] # Pitch
        filtered_msg.orientation_covariance[8] = P_state[3,3] # Yaw
        
        self.pub.publish(filtered_msg)

if __name__ == '__main__':
    try:
        kf_node = KalmanFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass