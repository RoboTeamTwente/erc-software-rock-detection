#!/usr/bin/env python3
"""
ROS2 (Jazzy) rock detector node for the RealSense D435i.

Subscribes
  /camera/camera/color/image_raw                      (sensor_msgs/Image)
  /camera/camera/aligned_depth_to_color/image_raw     (sensor_msgs/Image)
  /camera/camera/aligned_depth_to_color/camera_info   (sensor_msgs/CameraInfo)

Publishes
  /rock_detector/image_annotated   sensor_msgs/Image   color image with boxes
  /rock_detector/detections        std_msgs/String     JSON list of kept rocks

The camera must be launched with align_depth.enable:=true.

Filtering (all tunable at runtime with `ros2 param set /rock_detector <name> <value>`):
  confidence, box size, and depth (rejects background such as trees and grass,
  where depth is missing or spread over a large range).

Run (after sourcing ROS2 and your python environment):
  python3 rock_detector_node.py --ros-args -p weights:=/path/to/best.pt

The size is a rough estimate: longest box side in pixels * distance / focal length.
It ignores rock tilt and box padding, so check it against a ruler before trusting it.
The depth filter numbers are starting guesses to tune with show_rejected:=true.
"""
import json
import time

import cv2
import message_filters
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


# Convert ROS image messages to arrays and calculate depth statistics.
def image_to_bgr(msg):
    """sensor_msgs/Image (rgb8 or bgr8) -> BGR uint8 array. No cv_bridge needed."""
    h, w = msg.height, msg.width
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported color encoding: {msg.encoding}")
    arr = np.frombuffer(msg.data, np.uint8).reshape(h, msg.step)[:, : w * 3].reshape(h, w, 3)
    return arr[:, :, ::-1].copy() if msg.encoding == "rgb8" else arr.copy()


def depth_to_meters(msg, depth_scale=0.001):
    """sensor_msgs/Image (16UC1 in mm, or 32FC1 in m) -> float32 meters, 0 = no depth."""
    h, w = msg.height, msg.width
    if msg.encoding in ("16UC1", "mono16"):
        arr = np.frombuffer(msg.data, np.uint16).reshape(h, msg.step // 2)[:, :w]
        return arr.astype(np.float32) * depth_scale
    if msg.encoding == "32FC1":
        arr = np.frombuffer(msg.data, np.float32).reshape(h, msg.step // 4)[:, :w]
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    raise ValueError(f"unsupported depth encoding: {msg.encoding}")


def bgr_to_msg(img, header):
    out = Image()
    out.header = header
    out.height, out.width = img.shape[:2]
    out.encoding = "bgr8"
    out.is_bigendian = 0
    out.step = img.shape[1] * 3
    out.data = img.tobytes()
    return out


def box_depth_stats(depth_m, box, shrink=0.25):
    """Depth statistics inside the central part of a box (x1, y1, x2, y2 in pixels)."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    dx, dy = int((x2 - x1) * shrink), int((y2 - y1) * shrink)
    roi = depth_m[max(0, y1 + dy):max(0, y2 - dy), max(0, x1 + dx):max(0, x2 - dx)]
    if roi.size == 0:
        return None
    valid = roi[roi > 0]
    valid_fraction = valid.size / roi.size
    if valid.size < 10:
        return {"valid_fraction": valid_fraction, "z": None, "spread": None}
    return {
        "valid_fraction": valid_fraction,
        "z": float(np.median(valid)),
        "spread": float(np.percentile(valid, 90) - np.percentile(valid, 10)),
    }


def keep_detection(box, conf, depth_m, img_w, img_h, p):
    """Return (keep, z, reason). p is a dict of parameters."""
    x1, y1, x2, y2 = box
    if conf < p["conf"]:
        return False, None, "low conf"
    if ((x2 - x1) * (y2 - y1)) / float(img_w * img_h) > p["max_area"]:
        return False, None, "box too large"
    st = box_depth_stats(depth_m, box)
    if st is None or st["z"] is None or st["valid_fraction"] < p["min_valid"]:
        return False, None, "no depth"
    z = st["z"]
    if not (p["min_z"] <= z <= p["max_z"]):
        return False, z, "out of range"
    if st["spread"] > max(p["min_spread"], p["spread_frac"] * z):
        return False, z, "depth spread"
    return True, z, "ok"


def detect_rocks(model, bgr, depth_m, intr, p):
    """Run the model and filter. intr = (fx, fy, cx, cy).
    Returns (kept, rejected, inference_ms)."""
    H, W = bgr.shape[:2]
    fx, fy, cx, cy = intr
    kwargs = dict(imgsz=int(p["imgsz"]), conf=0.10, verbose=False)
    if p.get("device"):
        kwargs["device"] = p["device"]
    t0 = time.time()
    res = model.predict(bgr, **kwargs)[0]
    infer_ms = (time.time() - t0) * 1000.0

    kept, rejected = [], []
    for b in res.boxes:
        box = [float(v) for v in b.xyxy[0].cpu().numpy().tolist()]
        conf = float(b.conf[0])
        ok, z, reason = keep_detection(box, conf, depth_m, W, H, p)
        if ok:
            u, v = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            kept.append({
                "conf": round(conf, 3),
                "bbox": [round(c, 1) for c in box],
                "distance_m": round(z, 3),
                "longest_side_cm": round(100.0 * max(box[2] - box[0], box[3] - box[1]) * z / fx, 1),
                "xyz_m": [round((u - cx) * z / fx, 3), round((v - cy) * z / fy, 3), round(z, 3)],
            })
        elif conf >= 0.10:
            rejected.append({"bbox": box, "conf": conf, "reason": reason})
    return kept, rejected, infer_ms


# Draw accepted detections in green and rejected detections in red.
def draw(bgr, kept, rejected, p, infer_ms):
    view = bgr.copy()
    for r in kept:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(view, f"{r['conf']:.2f} {r['distance_m']:.2f}m ~{r['longest_side_cm']:.0f}cm",
                    (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    if p["show_rejected"]:
        for r in rejected:
            x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(view, r["reason"], (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(view, f"rocks kept: {len(kept)}  conf>={p['conf']:.2f}  {infer_ms:.0f} ms",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return view


class RockDetector(Node):
    # These values can be overridden with ROS parameters.
    PARAM_DEFAULTS = {
        "weights": "",
        "device": "",
        "imgsz": 640,
        "conf": 0.40,
        "max_area": 0.25,
        "min_valid": 0.5,
        "min_z": 0.3,
        "max_z": 3.5,
        "min_spread": 0.10,
        "spread_frac": 0.20,
        "max_rate_hz": 5.0,
        "depth_scale": 0.001,
        "show_rejected": False,
        "publish_annotated": True,
        "color_topic": "/camera/camera/color/image_raw",
        "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
        "info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
    }

    def __init__(self):
        super().__init__("rock_detector")
        for name, default in self.PARAM_DEFAULTS.items():
            self.declare_parameter(name, default)

        weights = self.get_parameter("weights").value
        if not weights:
            raise RuntimeError("Set the weights parameter: -p weights:=/path/to/best.pt")
        self.model = YOLO(weights)
        self.get_logger().info(f"Loaded model: {weights}")

        self.intr = None
        self.last_t = 0.0
        self.warned_shape = False

        self.create_subscription(CameraInfo, self.get_parameter("info_topic").value,
                                 self.on_info, qos_profile_sensor_data)
        color_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("color_topic").value, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("depth_topic").value, qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 5, 0.05)
        self.sync.registerCallback(self.on_frames)

        self.pub_img = self.create_publisher(Image, "~/image_annotated", 1)
        self.pub_json = self.create_publisher(String, "~/detections", 10)
        self.get_logger().info("Waiting for camera frames. Is the camera running with align_depth.enable:=true ?")

    def params(self):
        return {name: self.get_parameter(name).value for name in self.PARAM_DEFAULTS}

    def on_info(self, msg):
        k = msg.k
        self.intr = (k[0], k[4], k[2], k[5])

    def on_frames(self, color_msg, depth_msg):
        p = self.params()
        now = time.monotonic()
        if now - self.last_t < 1.0 / max(float(p["max_rate_hz"]), 0.1):
            return
        if self.intr is None:
            self.get_logger().warn("No camera_info yet, skipping frame.", throttle_duration_sec=5.0)
            return
        self.last_t = now

        try:
            bgr = image_to_bgr(color_msg)
            depth_m = depth_to_meters(depth_msg, float(p["depth_scale"]))
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        if bgr.shape[:2] != depth_m.shape:
            if not self.warned_shape:
                self.get_logger().error(
                    f"Color {bgr.shape[:2]} and depth {depth_m.shape} sizes differ. "
                    "Use the aligned_depth_to_color topic (align_depth.enable:=true).")
                self.warned_shape = True
            return

        kept, rejected, infer_ms = detect_rocks(self.model, bgr, depth_m, self.intr, p)

        out = {
            "stamp": color_msg.header.stamp.sec + color_msg.header.stamp.nanosec * 1e-9,
            "frame_id": color_msg.header.frame_id,
            "rocks": kept,
        }
        self.pub_json.publish(String(data=json.dumps(out)))
        if p["publish_annotated"]:
            self.pub_img.publish(bgr_to_msg(draw(bgr, kept, rejected, p, infer_ms), color_msg.header))
        self.get_logger().info(f"rocks kept: {len(kept)}, rejected: {len(rejected)}, {infer_ms:.0f} ms",
                               throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = RockDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
