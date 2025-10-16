#!/usr/bin/env python3
import rclpy, torch, cv2, numpy as np, sys, os, time
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Int8
from nit_messages.msg import Detection, Detections
from nit_messages.msg import BoundingBox, BoundingBoxes
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
from utils.plots import plot_one_box, colors

IMAGE_WIDTH, IMAGE_HEIGHT = 1241, 376

class CubeDetectorROS2(Node):
    def __init__(self):
        super().__init__('cube_detector_rotation', namespace='rosa')
        self.bridge = CvBridge()
        self.device = select_device('')
        self.get_logger().info(f'Using device: {self.device}')
        self.half = self.device.type != 'cpu'
        self.grab_toggle = 0

        # load model
        weights = os.path.join(os.path.dirname(__file__), '640m_new_classes.pt')
        self.model = attempt_load(weights, map_location=self.device)
        self.imgsz = check_img_size(640, s=self.model.stride.max())
        if self.half: self.model.half()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))
        self.get_logger().info(f'Loaded model with {len(self.names)} classes.')

        # pubs/subs
        self.sub = self.create_subscription(Image, 'rgb/image_raw', self.image_callback, 10)
        self.pub_dets = self.create_publisher(Detections, 'detections', 10)
        self.pub_img = self.create_publisher(Image, 'annotated_image', 10)
        self.pub_grab = self.create_publisher(Int8, 'cube_hand', 10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.detect(frame, msg.header.stamp)

    def detect(self, im0, stamp):
        # --- Preprocess (same as ROS1) ---
        img = letterbox(im0, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # [1,3,H,W]

        conf_thres = 0.75
        iou_thres = 0.45

        # --- Inference ---
        with torch.no_grad():
            pred = self.model(img, augment=False)[0] 
            if pred.ndimension() == 2:
                pred = pred.unsqueeze(0)  # -> [1, N, C]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

        dets = []
        grab_state = 0

        # --- Postprocess & publish ---
        for det in pred:  # det: [M, 6 or 7] = [x1,y1,x2,y2,conf,cls,(angle)]
            if len(det):
                # map boxes back to original image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # loop detections; angle may or may not be present
                for row in reversed(det):
                    row = row.tolist()
                    if len(row) >= 7:
                        x1, y1, x2, y2, conf, cls, angle = row[:7]
                    else:
                        x1, y1, x2, y2, conf, cls = row[:6]
                        angle = 0.0

                    # center
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)

                    # message
                    det_msg = Detection()
                    det_msg.letter = str(self.names[int(cls)]) if int(cls) < len(self.names) else str(int(cls))
                    det_msg.color = "" 
                    det_msg.conf.data = float(conf)
                    det_msg.pose.x = float(cx)
                    det_msg.pose.y = float(cy)
                    det_msg.pose.theta = float(angle)
                    dets.append(det_msg)
                    c = int(cls) 
                    try:
                        plot_one_box([x1, y1, x2, y2], angle, im0,
                                    label=f"{det_msg.letter} {conf:.2f}",
                                    color=colors(c, True), line_thickness=2)
                    except TypeError:
                        cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    if y2 > 700 and self.grab_toggle == 0:
                        grab_state = 1
                        self.grab_toggle = 1
                    else:
                        self.grab_toggle = 0

        # Publish detections
        msg = Detections()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.detections = dets
        self.pub_dets.publish(msg)

        # Publish grab Int8
        self.pub_grab.publish(Int8(data=int(grab_state)))

        # Publish annotated image
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(im0, encoding='bgr8'))

def main():
    rclpy.init()
    node = CubeDetectorROS2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
