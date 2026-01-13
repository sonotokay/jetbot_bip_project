#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class KalmanFilterNode:
    def __init__(self):
        # 1. Initialize ROS Node
        rospy.init_node('jetbot_kalman_filter', anonymous=True)

        # 2. Get Parameters (Optimized for Launch File)
        # Use private parameters (~) so they can be set via <param> in launch file
        self.imu_topic = rospy.get_param('~input_topic', '/imu')
        self.pub_topic = rospy.get_param('~output_topic', '/kalman_imu')
        
        # Tuning Defaults (can be overridden by launch file)
        # Gyro Noise: Lower = trust gyro more for yaw/smoothness
        self.default_gyro_var = rospy.get_param('~default_gyro_var', 0.005) 
        # Accel Noise: Lower = trust accel more for roll/pitch stability
        self.default_accel_var = rospy.get_param('~default_accel_var', 0.05)
        
        # 3. State Initialization
        # State Vector: [qw, qx, qy, qz] (Quaternion)
        self.x_est = np.array([[1., 0., 0., 0.]]).T 
        self.P_est = np.eye(4) * 0.1 # Initial uncertainty covariance

        self.C = np.eye(4) # Observation matrix (Identity)
        self.last_time = None

        # 4. Setup Publisher and Subscriber
        self.pub = rospy.Publisher(self.pub_topic, Imu, queue_size=10)
        self.sub = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
        
        rospy.loginfo(f"Kalman Filter started. Listening: {self.imu_topic} -> Publishing: {self.pub_topic}")
        rospy.loginfo(f"Defaults -> Gyro Var: {self.default_gyro_var}, Accel Var: {self.default_accel_var}")

    def imu_callback(self, msg):
        # --- Time Delta Calculation ---
        current_time = msg.header.stamp.to_sec()

        if self.last_time is None:
            self.last_time = current_time
            return

        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Safety check for invalid time jumps
        if dt <= 0: return

        # --- 1. HANDLE COVARIANCE (The Fix) ---
        # Get raw variances from the incoming message (average of diagonals)
        raw_gyro_var = (msg.angular_velocity_covariance[0] + 
                        msg.angular_velocity_covariance[4] + 
                        msg.angular_velocity_covariance[8]) / 3.0
        
        raw_accel_var = (msg.linear_acceleration_covariance[0] + 
                         msg.linear_acceleration_covariance[4] + 
                         msg.linear_acceleration_covariance[8]) / 3.0

        # Logic: If covariance is "Unknown/Garbage" (e.g. > 1.0 or 0.0), use our defaults
        if raw_gyro_var > 1.0 or raw_gyro_var == 0.0:
            gyro_var = self.default_gyro_var
        else:
            gyro_var = raw_gyro_var

        if raw_accel_var > 1.0 or raw_accel_var == 0.0:
            accel_var = self.default_accel_var
        else:
            accel_var = raw_accel_var

        # Construct Dynamic Process Noise (Q) and Measurement Noise (R) matrices
        Q_scale = gyro_var * (dt / 2.0)**2
        process_noise_covar = np.eye(4) * Q_scale

        # Convert accel variance (m/s^2) to approximate radian variance
        R_scale = accel_var / (9.81**2) 
        measurement_noise_covar = np.eye(4) * R_scale

        # --- 2. EXTRACT RAW SENSOR DATA ---
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        gx, gy, gz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z

        # --- 3. PREDICTION STEP (Integrate Gyroscope) ---
        # Quaternion derivative matrix based on angular velocity
        Un = np.array([
            [0, -gx, -gy, -gz],
            [gx,  0,  gz, -gy],
            [gy, -gz,  0,  gx],
            [gz,  gy, -gx,  0]
        ])
        
        # Update State Transition Matrix A
        A = np.eye(4) + (0.5 * dt * Un)
        
        # Predict State and Error Covariance
        x_pred = A @ self.x_est
        P_pred = A @ self.P_est @ A.T + process_noise_covar
        
        # Normalize predicted quaternion to maintain unit length
        x_pred = x_pred / np.linalg.norm(x_pred)

        # --- 4. MEASUREMENT STEP (Correct with Accelerometer) ---
        # Calculate Roll and Pitch from Gravity Vector
        meas_roll = np.arctan2(ay, az)
        # Sqrt protection added for pitch calculation
        meas_pitch = np.arctan2(-ax, np.sqrt(ay*ay + az*az) + 1e-9)
        
        # --- RELATIVE YAW LOGIC ---
        # Since we have NO Magnetometer, we cannot measure Yaw absolute.
        # We take the Yaw from our PREDICTED state (x_pred) and use it for the measurement.
        # This effectively tells the filter: "Trust the Gyro 100% for Yaw."
        
        # Convert prediction to Euler to extract current Yaw
        current_quat_list = [x_pred[1,0], x_pred[2,0], x_pred[3,0], x_pred[0,0]] # [x,y,z,w]
        _, _, pred_yaw = euler_from_quaternion(current_quat_list)

        # Reconstruct a "Measurement Quaternion" using Accel Roll/Pitch + Predicted Yaw
        q_meas_tf = quaternion_from_euler(meas_roll, meas_pitch, pred_yaw)
        
        # Convert back to our state format [w, x, y, z]
        measured_quaternion = np.array([[q_meas_tf[3], q_meas_tf[0], q_meas_tf[1], q_meas_tf[2]]]).T

        # Fix quaternion sign ambiguity (ensure dot product is positive)
        if np.dot(x_pred.T, measured_quaternion) < 0:
            measured_quaternion = -measured_quaternion

        # --- 5. UPDATE STEP (Kalman Gain) ---
        # Calculate Kalman Gain
        S = self.C @ P_pred @ self.C.T + measurement_noise_covar
        K_gain = P_pred @ self.C.T @ np.linalg.inv(S)

        # Correct the predicted state
        y_residual = measured_quaternion - (self.C @ x_pred)
        x_new = x_pred + (K_gain @ y_residual)
        
        # Update Error Covariance
        P_new = (np.eye(4) - K_gain @ self.C) @ P_pred

        # Normalize final result
        x_new = x_new / np.linalg.norm(x_new)
        
        # Store for next iteration
        self.x_est = x_new
        self.P_est = P_new

        # --- 6. PUBLISH RESULT ---
        self.publish_filtered_msg(msg, x_new, P_new)

    def publish_filtered_msg(self, original_msg, x_state, P_state):
        filtered_msg = Imu()
        filtered_msg.header = original_msg.header # Keep original timestamp/frame_id
        
        # Map state [w, x, y, z] to ROS message
        filtered_msg.orientation.w = x_state[0, 0]
        filtered_msg.orientation.x = x_state[1, 0]
        filtered_msg.orientation.y = x_state[2, 0]
        filtered_msg.orientation.z = x_state[3, 0]
        
        # Pass through original velocity/acceleration
        filtered_msg.angular_velocity = original_msg.angular_velocity
        filtered_msg.angular_velocity_covariance = original_msg.angular_velocity_covariance
        filtered_msg.linear_acceleration = original_msg.linear_acceleration
        filtered_msg.linear_acceleration_covariance = original_msg.linear_acceleration_covariance
        
        # Populate Orientation Covariance (Approximation)
        # We use the diagonal of the P matrix to give downstream nodes an idea of confidence
        filtered_msg.orientation_covariance = [0.0] * 9
        filtered_msg.orientation_covariance[0] = P_state[1,1] # Roll var
        filtered_msg.orientation_covariance[4] = P_state[2,2] # Pitch var
        filtered_msg.orientation_covariance[8] = P_state[3,3] # Yaw var
        
        self.pub.publish(filtered_msg)

if __name__ == '__main__':
    try:
        kf_node = KalmanFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass