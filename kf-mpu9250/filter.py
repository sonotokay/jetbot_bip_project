#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class KalmanFilterNode:
    def __init__(self):
        rospy.init_node('jetbot_kalman_filter', anonymous=True)

        self.imu_topic = "/imu"
        self.pub_topic = "/kalman_imu"
        
        # Initial State [w, x, y, z]
        self.x_est = np.array([[1., 0., 0., 0.]]).T 
        self.P_est = np.eye(4) * 0.1 

        self.C = np.eye(4) 
        self.last_time = None

        self.pub = rospy.Publisher(self.pub_topic, Imu, queue_size=10)
        self.sub = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
        
        rospy.loginfo("Kalman Filter running. Filtering Roll/Pitch (Accel+Gyro) and Smoothing Yaw (Gyro only).")

    def imu_callback(self, msg):
        current_time = msg.header.stamp.to_sec()

        if self.last_time is None:
            self.last_time = current_time
            return

        dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt <= 0: return

        # --- 1. HANDLE COVARIANCE (THE FIX) ---
        # Get raw variances from the message
        raw_gyro_var = (msg.angular_velocity_covariance[0] + msg.angular_velocity_covariance[4] + msg.angular_velocity_covariance[8]) / 3.0
        raw_accel_var = (msg.linear_acceleration_covariance[0] + msg.linear_acceleration_covariance[4] + msg.linear_acceleration_covariance[8]) / 3.0

        # FIX: Check if covariance is "Garbage/Unknown" (e.g. 1000000.0) or zero
        # If so, override with realistic sensor noise values.
        # Gyro Noise: ~0.005 to 0.01 is typical for MPUs
        if raw_gyro_var > 1.0 or raw_gyro_var == 0:
            gyro_var = 0.005 
        else:
            gyro_var = raw_gyro_var

        # Accel Noise: ~0.01 to 0.1 is typical
        if raw_accel_var > 1.0 or raw_accel_var == 0:
            accel_var = 0.05
        else:
            accel_var = raw_accel_var

        # Calculate Q and R matrices based on these fixed variances
        Q_scale = gyro_var * (dt / 2.0)**2
        process_noise_covar = np.eye(4) * Q_scale

        R_scale = accel_var / (9.81**2) # Convert accel noise to approximate radian noise
        measurement_noise_covar = np.eye(4) * R_scale

        # --- 2. EXTRACT DATA ---
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        gx, gy, gz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z

        # --- 3. PREDICTION (Gyro Step) ---
        Un = np.array([
            [0, -gx, -gy, -gz],
            [gx,  0,  gz, -gy],
            [gy, -gz,  0,  gx],
            [gz,  gy, -gx,  0]
        ])
        
        A = np.eye(4) + (0.5 * dt * Un)
        x_pred = A @ self.x_est
        P_pred = A @ self.P_est @ A.T + process_noise_covar
        x_pred = x_pred / np.linalg.norm(x_pred)

        # --- 4. MEASUREMENT (Accel Step) ---
        meas_roll = np.arctan2(ay, az)
        meas_pitch = np.arctan2(-ax, np.sqrt(ay*ay + az*az))
        
        # YAW LOGIC: 
        # Since we have NO Magnetometer, we use the PREDICTED Yaw.
        # This tells the filter: "Trust the Gyro for Yaw completely."
        current_quat_list = [x_pred[1,0], x_pred[2,0], x_pred[3,0], x_pred[0,0]] 
        _, _, pred_yaw = euler_from_quaternion(current_quat_list)

        q_meas_tf = quaternion_from_euler(meas_roll, meas_pitch, pred_yaw)
        measured_quaternion = np.array([[q_meas_tf[3], q_meas_tf[0], q_meas_tf[1], q_meas_tf[2]]]).T

        if np.dot(x_pred.T, measured_quaternion) < 0:
            measured_quaternion = -measured_quaternion

        # --- 5. UPDATE (Kalman Gain) ---
        S = self.C @ P_pred @ self.C.T + measurement_noise_covar
        K_gain = P_pred @ self.C.T @ np.linalg.inv(S)

        y_residual = measured_quaternion - (self.C @ x_pred)
        x_new = x_pred + (K_gain @ y_residual)
        P_new = (np.eye(4) - K_gain @ self.C) @ P_pred

        x_new = x_new / np.linalg.norm(x_new)
        
        self.x_est = x_new
        self.P_est = P_new

        # --- 6. PUBLISH ---
        self.publish_filtered_msg(msg, x_new, P_new)

    def publish_filtered_msg(self, original_msg, x_state, P_state):
        filtered_msg = Imu()
        filtered_msg.header = original_msg.header
        
        filtered_msg.orientation.w = x_state[0, 0]
        filtered_msg.orientation.x = x_state[1, 0]
        filtered_msg.orientation.y = x_state[2, 0]
        filtered_msg.orientation.z = x_state[3, 0]
        
        filtered_msg.angular_velocity = original_msg.angular_velocity
        filtered_msg.angular_velocity_covariance = original_msg.angular_velocity_covariance
        filtered_msg.linear_acceleration = original_msg.linear_acceleration
        filtered_msg.linear_acceleration_covariance = original_msg.linear_acceleration_covariance
        
        # We fill covariance to show we are confident in the filter
        filtered_msg.orientation_covariance = [0.0] * 9
        filtered_msg.orientation_covariance[0] = P_state[1,1]
        filtered_msg.orientation_covariance[4] = P_state[2,2]
        filtered_msg.orientation_covariance[8] = P_state[3,3]
        
        self.pub.publish(filtered_msg)

if __name__ == '__main__':
    try:
        kf_node = KalmanFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass