import math
import threading
import numpy as np
import threading
from scipy.spatial.transform import Rotation

import rclpy
import tf2_ros
from rclpy.node import Node
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo, JointState, Imu, LaserScan

from discoverse.robots_env.mmk2_base import MMK2Base, MMK2Cfg
from discoverse.utils import PIDarray, camera2k, get_site_tmat

# ---- headless/test robustness: never let a GLFW window close the sim ----
# discoverse's simulator render loop does:
#     if glfw.window_should_close(self.window): self.running = False
# Under WSLg / flaky rendering contexts the window can report "should close"
# spuriously, which silently ends the whole simulation (the client keeps
# waiting while the server exits - verified: server Exited(0) right after
# "referee results saved", client logs stale odom for the rest of the run).
# We patch window_should_close BEFORE discoverse's renderer ever polls it, so
# the physics loop only stops on rclpy.ok() (Ctrl-C / ROS shutdown).
try:
    import glfw as _glfw
    _glfw_should_close_orig = _glfw.window_should_close
    _glfw.window_should_close = lambda _window: False
except Exception as _exc:  # pragma: no cover - windowless server
    _glfw_should_close_orig = None

class MMK2ROS2(MMK2Base, Node):
    target_control = np.zeros(19)
    def __init__(self, config: MMK2Cfg):
        self.tctr_base = self.target_control[:2]
        self.tctr_slide = self.target_control[2:3]
        self.tctr_head = self.target_control[3:5]
        self.tctr_left_arm = self.target_control[5:11]
        self.tctr_lft_gripper = self.target_control[11:12]
        self.tctr_right_arm = self.target_control[12:18]
        self.tctr_rgt_gripper = self.target_control[18:19]

        super().__init__(config)
        Node.__init__(self, 'MMK2_mujoco_node')
        # MuJoCo's MjData is not thread-safe.  The physics loop mutates it
        # while the optional lidar wrapper traces rays against it, so those
        # two operations must never run concurrently.
        self._physics_lock = threading.RLock()

        self.pid_base_vel = PIDarray(
            kps=np.array([ 7.5 ,  7.5 ]),
            kis=np.array([  .0 ,   .0 ]),
            kds=np.array([  .0 ,   .0 ]),
            integrator_maxs=np.array([5.0, 5.0]),
        )

        self.init_topic_publisher()
        self.init_topic_subscriber()

        if self.config.lidar_s2_sim:
            self._init_lidar_sensor()
            if not getattr(self, "_lidar_api", None):
                self.config.lidar_s2_sim = False

    def _init_lidar_sensor(self):
        """Initialize simulated lidar with compatibility fallbacks.

        Different official images expose slightly different mujoco_lidar
        signatures. We try the named form first, then positional forms, and
        finally fall back to a self-contained mj_ray implementation; if all
        fail the lidar is disabled gracefully.
        """
        self.lidar_frame_id = "laser"
        try:
            from mujoco_lidar.lidar_wrapper import MjLidarWrapper
            from mujoco_lidar.scan_gen import create_lidar_single_line
        except Exception as exc:
            self.get_logger().warn("[server] mujoco_lidar import failed: %s" % exc)
            self.lidar_s2 = None
            return

        self.rays_theta, self.rays_phi = create_lidar_single_line(360, np.pi * 2.0)

        wrapper = None
        self._lidar_api = None
        init_errors = []
        # The V2 server image exposes the current mujoco_lidar API:
        # MjLidarWrapper(model, site_name, backend=...) plus trace_rays().
        # Keep the legacy call forms after it for older training images.
        for args, kwargs, api in (
            ((self.mj_model, self.lidar_frame_id), {"backend": "cpu"}, "trace_rays"),
            ((self.mj_model, self.mj_data), {"site_name": self.lidar_frame_id}, "get_lidar_points"),
            ((self.mj_model, self.mj_data, self.lidar_frame_id), {}, "get_lidar_points"),
        ):
            try:
                wrapper = MjLidarWrapper(*args, **kwargs)
                self._lidar_api = api
                break
            except TypeError as exc:
                init_errors.append(str(exc))
            except Exception as exc:
                init_errors.append(str(exc))
                break

        if wrapper is None:
            self.get_logger().warn(
                "[server] lidar init failed; fallback to depth-camera safety only: %s"
                % "; ".join(init_errors[-2:])
            )
            self.lidar_s2 = None
            return

        self.lidar_s2 = wrapper
        # Sanity-check one trace from the open start pocket. The CPU backend
        # of some mujoco_lidar builds constructs fine but returns distances of
        # a few centimetres for every ray (verified: max 0.024 m in an open
        # arena), which silently disables every client-side avoidance sector.
        # Fall back to our own mj_ray implementation in that case.
        healthy = False
        with self._physics_lock:
            try:
                if self._lidar_api == "trace_rays":
                    probe = np.asarray(
                        self.lidar_s2.trace_rays(self.mj_data, self.rays_theta, self.rays_phi),
                        dtype=float,
                    )
                else:
                    probe = np.asarray(
                        self.lidar_s2.get_lidar_points(self.rays_phi, self.rays_theta, self.mj_data),
                        dtype=float,
                    )
                healthy = bool(np.nanmax(probe) > 1.0)
            except Exception as exc:
                self.get_logger().warn("[server] lidar probe trace failed: %s" % exc)
        if healthy:
            self.get_logger().info("[server] lidar wrapper trace verified (max=%.1f m)" % float(np.nanmax(probe)))
        else:
            self.get_logger().warn(
                "[server] lidar wrapper returned unusable ranges; "
                "falling back to the built-in mj_ray scanner")
            self.lidar_s2 = None
            self._lidar_api = "mj_ray"
            import mujoco
            self._mj_ray_chassis_body = None
            # mj_ray's bodyexclude does NOT cover the subtree (verified), so
            # the chassis body itself (agv_link) must be excluded: the lidar
            # site sits on the chassis and otherwise every ray self-hits at
            # ~0.02 m, killing all avoidance sectors.
            for name in ("agv_link", "mmk2"):
                try:
                    self._mj_ray_chassis_body = int(mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name))
                    break
                except Exception:
                    continue

        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_static_transform(header_frame_id='base_link', child_frame_id=self.lidar_frame_id)

    def _scan_with_mj_ray(self):
        """CPU lidar fallback: cast 360 rays with MuJoCo's mj_ray.

        Returns distances aligned with ``self.rays_theta`` (no hits are -1.0).

        Only geom group 0 is cast against: the scene's collision geoms
        (shelves, walls, table, boxes) are all group 0, while the robot's own
        collision geoms are group 4 and its visual meshes are non-zero groups.
        ``bodyexclude=agv_link`` alone still left the ARM links in the cloud at
        0.3-1.0 m (the start-pose front sector read 0.98 m in an open aisle),
        which enclosed the A* start cell with self-hits.
        """
        import mujoco

        site = self.mj_data.site(self.lidar_frame_id)
        pnt = site.xpos.copy()
        base_mat = np.asarray(self.mj_data.body("agv_link").xmat, dtype=float).reshape(3, 3)
        yaw = math.atan2(base_mat[1, 0], base_mat[0, 0])
        dists = np.empty(len(self.rays_theta), dtype=float)
        exclude = getattr(self, "_mj_ray_chassis_body", -1) or -1
        # Only geom group 0 (the scene's collision geoms). The robot's own
        # collision geoms are group 4 and its visual meshes are non-zero
        # groups, so this removes the carried arm from the cloud; the binding
        # wants a 6-byte mask like mjvOption, not a scalar bitmask.
        group_mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        for index, theta in enumerate(self.rays_theta):
            angle = yaw + float(theta)
            vec = np.array([math.cos(angle), math.sin(angle), 0.0], dtype=float)
            dist = mujoco.mj_ray(
                self.mj_model, self.mj_data, pnt, vec,
                geomgroup=group_mask, flg_static=True, bodyexclude=exclude,
                geomid=None,
            )
            dists[index] = -1.0 if dist is None or dist < 0 else float(dist)
        return dists

    def init_topic_publisher(self):
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # rostopic joint state
        self.joint_state_puber = self.create_publisher(JointState, '/joint_states', 5)
        self.joint_state = JointState()
        self.joint_state.name = [
            "slide_joint", "head_yaw_joint", "head_pitch_joint",
            "left_arm_joint1" , "left_arm_joint2" , "left_arm_joint3" , "left_arm_joint4" , "left_arm_joint5" , "left_arm_joint6" , "left_arm_eef_gripper_joint" ,
            "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_eef_gripper_joint",
        ]
        self.joint_state.position = self.sensor_qpos[2:].tolist()
        self.joint_state.velocity = self.sensor_qvel[2:].tolist()
        self.joint_state.effort = self.sensor_force[2:].tolist()

        # rostopic imu
        # # # self.imu_puber = self.create_publisher(Imu, '/imu', 5)
        # # self.imu_msg = Imu()
        # # self.imu_msg.header.frame_id = "mmk2_imu_link"

        # rostopic odometry
        self.odom_puber = self.create_publisher(Odometry, '/slamware_ros_sdk_server_node/odom', 5)
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "/odom"
        # self.odom_msg.child_frame_id = "base_link"

        # image
        self.bridge = CvBridge()

        # lidar
        if self.config.lidar_s2_sim:
            self.lidar_s2_puber = self.create_publisher(LaserScan, '/slamware_ros_sdk_server_node/scan', 1)

        # image publisher, camera info publisher,  Initialize camera info messages
        if 0 in self.config.obs_rgb_cam_id:
            self.head_color_puber  = self.create_publisher(Image, '/head_camera/color/image_raw', 2)
            self.head_color_info_puber  = self.create_publisher(CameraInfo, '/head_camera/color/camera_info', 2)
            self.head_color_info = CameraInfo()
            self.head_color_info.width = self.config.render_set["width"]
            self.head_color_info.height = self.config.render_set["height"]
            self.head_color_info.k = camera2k(self.mj_model.cam_fovy[0] * np.pi / 180., self.config.render_set["width"], self.config.render_set["height"]).flatten().tolist()

        if 1 in self.config.obs_rgb_cam_id:
            self.left_color_puber  = self.create_publisher(Image, '/left_camera/color/image_raw', 2)
            self.left_color_info_puber  = self.create_publisher(CameraInfo, '/left_camera/color/camera_info', 2)
            self.left_color_info = CameraInfo()
            self.left_color_info.width = self.config.render_set["width"]
            self.left_color_info.height = self.config.render_set["height"]
            self.left_color_info.k = camera2k(self.mj_model.cam_fovy[1] * np.pi / 180., self.config.render_set["width"], self.config.render_set["height"]).flatten().tolist()

        if 2 in self.config.obs_rgb_cam_id:
            self.right_color_puber = self.create_publisher(Image, '/right_camera/color/image_raw', 2)
            self.right_color_info_puber = self.create_publisher(CameraInfo, '/right_camera/color/camera_info', 2)
            self.right_color_info = CameraInfo()
            self.right_color_info.width = self.config.render_set["width"]
            self.right_color_info.height = self.config.render_set["height"]
            self.right_color_info.k = camera2k(self.mj_model.cam_fovy[2] * np.pi / 180., self.config.render_set["width"], self.config.render_set["height"]).flatten().tolist()

        if 0 in self.config.obs_depth_cam_id:
            self.head_depth_puber  = self.create_publisher(Image, '/head_camera/aligned_depth_to_color/image_raw', 2)
            self.head_depth_info_puber  = self.create_publisher(CameraInfo, '/head_camera/aligned_depth_to_color/camera_info', 2)
            self.head_depth_info = CameraInfo()
            self.head_depth_info.width = self.config.render_set["width"]
            self.head_depth_info.height = self.config.render_set["height"]
            self.head_depth_info.k = camera2k(self.mj_model.cam_fovy[0] * np.pi / 180., self.config.render_set["width"], self.config.render_set["height"]).flatten().tolist()
 
        if (self.config.obs_rgb_cam_id is not None) or (self.config.obs_depth_cam_id is not None):            
            # Publish camera info periodically
            self.create_timer(1.0, self.publish_camera_info)

    def init_topic_subscriber(self):
        self.cmd_vel_suber = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 5)
        self.spine_cmd_suber = self.create_subscription(Float64MultiArray, '/spine_forward_position_controller/commands', self.cmd_spine_callback, 5)
        self.head_cmd_suber = self.create_subscription(Float64MultiArray, '/head_forward_position_controller/commands', self.cmd_head_callback, 5)
        self.left_arm_cmd_suber = self.create_subscription(Float64MultiArray, '/left_arm_forward_position_controller/commands', self.cmd_left_arm_callback, 5)
        self.right_arm_cmd_suber = self.create_subscription(Float64MultiArray, '/right_arm_forward_position_controller/commands', self.cmd_right_arm_callback, 5)

    def publish_camera_info(self):
        if 0 in self.config.obs_rgb_cam_id:
            self.head_color_info_puber.publish(self.head_color_info)
        if 1 in self.config.obs_rgb_cam_id:
            self.left_color_info_puber.publish(self.left_color_info)
        if 2 in self.config.obs_rgb_cam_id:
            self.right_color_info_puber.publish(self.right_color_info)
        if 0 in self.config.obs_depth_cam_id:
            self.head_depth_info_puber.publish(self.head_depth_info)

    def cmd_vel_callback(self, msg: Twist):
        self.tctr_base[0] = (msg.linear.x - msg.angular.z * self.wheel_distance) / self.wheel_radius
        self.tctr_base[1] = (msg.linear.x + msg.angular.z * self.wheel_distance) / self.wheel_radius

    def cmd_spine_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 1:
            self.tctr_slide[:] = msg.data[:]
        else:
            self.get_logger().error('Spine command length error')

    def cmd_head_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 2:
            self.tctr_head[:] = msg.data[:]
        else:
            self.get_logger().error("head command length error")

    def cmd_left_arm_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.tctr_left_arm[:] = msg.data[:6]
            self.tctr_lft_gripper[:] = msg.data[6:]
        else:
            self.get_logger().error("left arm command length error")

    def cmd_right_arm_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.tctr_right_arm[:] = msg.data[:6]
            self.tctr_rgt_gripper[:] = msg.data[6:]
        else:
            self.get_logger().error("right arm command length error")

    def resetState(self):
        super().resetState()
        self.pid_base_vel.reset()
        self.target_control[:] = self.init_joint_ctrl[:]

    def updateControl(self, action):
        wheel_force = self.pid_base_vel.output(np.clip(self.tctr_base - self.sensor_wheel_qvel, -2.5, 2.5), self.mj_model.opt.timestep)
        self.mj_data.ctrl[:2] = np.clip(wheel_force, self.mj_model.actuator_ctrlrange[:2,0], self.mj_model.actuator_ctrlrange[:2,1])
        self.mj_data.ctrl[2:self.njctrl] = np.clip(action[2:self.njctrl], self.mj_model.actuator_ctrlrange[2:self.njctrl,0], self.mj_model.actuator_ctrlrange[2:self.njctrl,1])

    def publish_static_transform(self, header_frame_id, child_frame_id):
        stfs_msg = TransformStamped()
        stfs_msg.header.stamp = self.get_clock().now().to_msg()
        stfs_msg.header.frame_id = header_frame_id
        stfs_msg.child_frame_id = child_frame_id

        tmat_base = get_site_tmat(self.mj_data, header_frame_id)
        tmat_child = get_site_tmat(self.mj_data, child_frame_id)
        tmat_trans = np.linalg.inv(tmat_base) @ tmat_child
        
        stfs_msg.transform.translation.x = tmat_trans[0, 3]
        stfs_msg.transform.translation.y = tmat_trans[1, 3]
        stfs_msg.transform.translation.z = tmat_trans[2, 3]

        quat = Rotation.from_matrix(tmat_trans[:3, :3]).as_quat()
        stfs_msg.transform.rotation.x = quat[0]
        stfs_msg.transform.rotation.y = quat[1]
        stfs_msg.transform.rotation.z = quat[2]
        stfs_msg.transform.rotation.w = quat[3]

        self.static_broadcaster.sendTransform(stfs_msg)

    def thread_publidartopic(self, freq=12):
        if not self.config.lidar_s2_sim:
            return
                      
        rate = self.create_rate(freq)
        while rclpy.ok():
            # The renderer can flip running=False once (see module docstring);
            # that must not permanently kill lidar publishing, or the client
            # sees a dead laser while the match is still running.
            if not self.running:
                self.running = True
            with self._physics_lock:
                if self._lidar_api == "mj_ray":
                    dists = self._scan_with_mj_ray()
                elif self._lidar_api == "trace_rays":
                    dists = np.asarray(
                        self.lidar_s2.trace_rays(self.mj_data, self.rays_theta, self.rays_phi),
                        dtype=float,
                    )
                else:
                    points = self.lidar_s2.get_lidar_points(self.rays_phi, self.rays_theta, self.mj_data)
                    dists = np.linalg.norm(points[:, :2], axis=1)
            scan_msg = LaserScan()
            scan_msg.header.frame_id = self.lidar_frame_id
            scan_msg.header.stamp = self.get_clock().now().to_msg()
            scan_msg.angle_min = float(np.pi)
            scan_msg.angle_max = float(-np.pi * 179. / 180.)
            scan_msg.angle_increment = float(-2. * np.pi / 360.)
            scan_msg.time_increment = 0.0
            # Official V2 interface metadata: 360 points at 12 Hz, min valid
            # range 0.02 m, max valid range 12 m. Report the advertised range
            # instead of the per-frame measured extents so client-side range
            # gates behave identically against the official server.
            scan_msg.range_min = 0.02
            scan_msg.range_max = 12.0
            scan_msg.ranges = np.where(dists < 0.02, float("inf"), dists)[::-1].astype(np.float32).tolist()
            scan_msg.intensities = []

            self.lidar_s2_puber.publish(scan_msg)
            rate.sleep()

    def physics_step(self):
        """Advance MuJoCo without racing the optional lidar tracing thread."""
        with self._physics_lock:
            self.step(self.target_control)

    def thread_pubros2topic(self, freq=30):
        rate = self.create_rate(freq)
        while rclpy.ok():
            # Same resilience as the lidar thread: a single running=False
            # (transient renderer hiccup) must not end odometry/joint
            # publishing for the rest of the match.
            if not self.running:
                self.running = True
            time_stamp = self.get_clock().now().to_msg()

            self.joint_state.header.stamp = time_stamp
            self.joint_state.position = self.sensor_qpos[2:].tolist()
            self.joint_state.velocity = self.sensor_qvel[2:].tolist()
            self.joint_state.effort = self.sensor_force[2:].tolist()
            self.joint_state_puber.publish(self.joint_state)

            self.odom_msg.header.stamp = time_stamp
            self.odom_msg.pose.pose.position.x = self.sensor_base_position[0]
            self.odom_msg.pose.pose.position.y = self.sensor_base_position[1]
            self.odom_msg.pose.pose.position.z = self.sensor_base_position[2]
            self.odom_msg.pose.pose.orientation.w = self.sensor_base_orientation[0]
            self.odom_msg.pose.pose.orientation.x = self.sensor_base_orientation[1]
            self.odom_msg.pose.pose.orientation.y = self.sensor_base_orientation[2]
            self.odom_msg.pose.pose.orientation.z = self.sensor_base_orientation[3]
            self.odom_msg.twist.twist.linear.x = self.sensor_base_linear_vel[0]
            self.odom_msg.twist.twist.linear.y = self.sensor_base_linear_vel[1]
            self.odom_msg.twist.twist.linear.z = self.sensor_base_linear_vel[2]
            self.odom_msg.twist.twist.angular.x = self.sensor_base_gyro[0]
            self.odom_msg.twist.twist.angular.y = self.sensor_base_gyro[1]
            self.odom_msg.twist.twist.angular.z = self.sensor_base_gyro[2]
            self.odom_puber.publish(self.odom_msg)
            
            trans_msg = TransformStamped()
            trans_msg.header.stamp = time_stamp
            trans_msg.header.frame_id = "odom"
            trans_msg.child_frame_id = "base_link"
            trans_msg.transform.translation.x = self.sensor_base_position[0]
            trans_msg.transform.translation.y = self.sensor_base_position[1]
            trans_msg.transform.translation.z = self.sensor_base_position[2]
            trans_msg.transform.rotation.w = self.sensor_base_orientation[0]
            trans_msg.transform.rotation.x = self.sensor_base_orientation[1]
            trans_msg.transform.rotation.y = self.sensor_base_orientation[2]
            trans_msg.transform.rotation.z = self.sensor_base_orientation[3]            
            self.tf_broadcaster.sendTransform(trans_msg)

            # self.imu_msg.header.stamp = time_stamp
            # self.imu_msg.orientation.w = self.sensor_base_orientation[0]
            # self.imu_msg.orientation.x = self.sensor_base_orientation[1]
            # self.imu_msg.orientation.y = self.sensor_base_orientation[2]
            # self.imu_msg.orientation.z = self.sensor_base_orientation[3]
            # self.imu_msg.angular_velocity.x = self.sensor_base_gyro[0]
            # self.imu_msg.angular_velocity.y = self.sensor_base_gyro[1]
            # self.imu_msg.angular_velocity.z = self.sensor_base_gyro[2]
            # self.imu_msg.linear_acceleration.x = self.sensor_base_acc[0]
            # self.imu_msg.linear_acceleration.y = self.sensor_base_acc[1]
            # self.imu_msg.linear_acceleration.z = self.sensor_base_acc[2]
            # # self.imu_puber.publish(self.imu_msg)

            obs_img = self.obs.get("img", {}) if isinstance(self.obs, dict) else {}
            obs_depth = self.obs.get("depth", {}) if isinstance(self.obs, dict) else {}

            if 0 in self.config.obs_rgb_cam_id and 0 in obs_img:
                head_color_img_msg = self.bridge.cv2_to_imgmsg(obs_img[0], encoding="rgb8")
                head_color_img_msg.header.stamp = time_stamp
                head_color_img_msg.header.frame_id = "head_camera"
                self.head_color_puber.publish(head_color_img_msg)

            if 1 in self.config.obs_rgb_cam_id and 1 in obs_img:
                left_color_img_msg = self.bridge.cv2_to_imgmsg(obs_img[1], encoding="rgb8")
                left_color_img_msg.header.stamp = time_stamp
                left_color_img_msg.header.frame_id = "left_camera"
                self.left_color_puber.publish(left_color_img_msg)

            if 2 in self.config.obs_rgb_cam_id and 2 in obs_img:
                right_color_img_msg = self.bridge.cv2_to_imgmsg(obs_img[2], encoding="rgb8")
                right_color_img_msg.header.stamp = time_stamp
                right_color_img_msg.header.frame_id = "right_camera"
                self.right_color_puber.publish(right_color_img_msg)

            if 0 in self.config.obs_depth_cam_id and 0 in obs_depth:
                head_depth_img = np.array(np.clip(obs_depth[0]*1e3, 0, 65535), dtype=np.uint16)
                head_depth_img_msg = self.bridge.cv2_to_imgmsg(head_depth_img, encoding="mono16")
                head_depth_img_msg.header.stamp = time_stamp
                head_depth_img_msg.header.frame_id = "head_camera"
                self.head_depth_puber.publish(head_depth_img_msg)

            rate.sleep()


if __name__ == "__main__":
    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = MMK2Cfg()
    cfg.mjcf_file_path = "mjcf/mmk2_floor.xml"
    cfg.use_gaussian_renderer = False
    cfg.obs_rgb_cam_id = [0,1,2]
    cfg.obs_depth_cam_id = [0]
    cfg.lidar_s2_sim = True
    cfg.render_set     = {
        "fps"    : 24,
        "width"  : 640,
        "height" : 480
    }

    exec_node = MMK2ROS2(cfg)
    exec_node.reset()

    spin_thread = threading.Thread(target=lambda:rclpy.spin(exec_node))
    spin_thread.start()

    publidar_thread = threading.Thread(target=exec_node.thread_publidartopic, args=(12,))
    publidar_thread.start()
    
    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(30,))
    pubtopic_thread.start()

    now_guard = 0
    loop_failures = 0

    # Belt and suspenders for headless verification: if the base ever flips
    # running=False for a reason other than rclpy shutdown (e.g. a rendering
    # hiccup that bypassed the glfw patch), keep the physics alive and say so.
    # The client depends on the server's odometry for its whole test window.
    while rclpy.ok():
        if not exec_node.running:
            exec_node.running = True
            if now_guard % 240 == 0:
                exec_node.get_logger().warn(
                    "[server] exec_node.running flipped False while ROS is up; "
                    "re-asserting and continuing physics (headless verification)")
            now_guard += 1
        try:
            exec_node.physics_step()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            # Same policy as the scoring server main loop: a transient
            # physics/render failure (mj_step already advanced before render)
            # must not kill the whole simulation.  Log rate-limited, keep
            # stepping; give up only on a pathological consecutive streak.
            loop_failures += 1
            if loop_failures == 1 or loop_failures % 240 == 0:
                exec_node.get_logger().warn(
                    "[server] physics step failed (%d consecutive): %r"
                    % (loop_failures, exc))
                if loop_failures <= 240:
                    import traceback
                    traceback.print_exc()
            if loop_failures >= 20000:
                exec_node.get_logger().error(
                    "[server] too many consecutive physics failures; ending run")
                break
            continue
        loop_failures = 0

    exec_node.destroy_node()
    rclpy.shutdown()
    publidar_thread.join()
    pubtopic_thread.join()
    spin_thread.join()
