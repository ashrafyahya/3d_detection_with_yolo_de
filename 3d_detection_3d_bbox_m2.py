# Dieses Skript unterscheidet sich vom 3d_detection_3d_bbox_m1.py mit durch die Tiefenrechnung.
# Hierbei wird die Tiefe aus Pixelntiefe berechnet.

import os
import random
import sys
import torch
import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO


def print_object_info(class_name, distance, height, width, depth, confidence):
    print("\n" + "-" * 40)
    print(f"Object name: {class_name}")
    print(f"Distance: {distance:.2f}m")
    print(f"Height: {height:.2f}m")
    print(f"Width: {width:.2f}m")
    print(f"Depth: {depth:.2f}m")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 40 + "\n")


def generate_random_color():
    return (random.randint(0, 255), (random.randint(0, 255)), (random.randint(0, 255)))


def draw_3d_bounding_box(image, points2d, color):
    #    4-------5
    #   /|      /|
    #  / |     / |
    # 0-------1  |
    # |  |    |  |
    # |  7----|--6
    # | /     | /
    # 3-------2
    faces = [
        [0, 1, 2, 3],  # Front
        [4, 5, 6, 7],  # Back
        [0, 1, 5, 4],  # Top
        [2, 3, 7, 6],  # Bottom
        [0, 3, 7, 4],  # Left
        [1, 2, 6, 5]   # Right
    ]
    
    overlay = image.copy()
    
    for face in faces:
        pts = np.array([points2d[i] for i in face], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
    
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    for i in range(4):
        cv2.line(image, points2d[i], points2d[(i + 1) % 4], color, 2)
        cv2.line(image, points2d[i + 4], points2d[(i + 1) % 4 + 4], color, 2)
        cv2.line(image, points2d[i], points2d[i + 4], color, 2)


def project_point(X, Y, Z, fx, fy, cx, cy):
    u = int((X * fx) / Z + cx)
    v = int((Y * fy) / Z + cy)
    return (u, v)


def draw_label_with_background(image, text, x, y, color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - h - 4), (x + w, y), bg_color, -1)
    cv2.putText(image, text, (x, y - 2), font, scale, color, thickness, cv2.LINE_AA)


def process_frame(frame, xyz_data, model, fx, fy, cx, cy, object_colors, device):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, verbose=False, device=device)
    annotated_frame = frame.copy()

    masks = results[0].masks.data if results[0].masks is not None else None

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name not in object_colors:
            object_colors[class_name] = generate_random_color()
        color = object_colors[class_name]

        # Maske oder Bounding Box auswerten
        if masks is not None:
            mask_resized = cv2.resize(masks[i].cpu().numpy(), (xyz_data.shape[1], xyz_data.shape[0]))
            mask_binary = (mask_resized > 0.5)
            masked_xyz = xyz_data[mask_binary][..., :3]

        else:
            box_xyz = xyz_data[y1:y2, x1:x2][..., :3].reshape(-1, 3)

            masked_xyz = box_xyz

        # Nur valide XYZ-Punkte behalten
        if masked_xyz.size == 0:
            continue

        z_vals = masked_xyz[:, 2]
        valid_mask = np.isfinite(z_vals) & (z_vals > 0) & (z_vals < 15)
        valid_xyz = masked_xyz[valid_mask]

        if valid_xyz.shape[0] == 0:
            continue

        # Mittelpunkt und Dimensionen berechnen
        centroid = np.median(valid_xyz, axis=0)
        min_vals = np.min(valid_xyz, axis=0)
        max_vals = np.max(valid_xyz, axis=0)

        width_m  = max_vals[0] - min_vals[0]  # X
        height_m = max_vals[1] - min_vals[1]  # Y
        depth_m  = max_vals[2] - min_vals[2]  # Z

        centroid = np.median(valid_xyz, axis=0)
        if centroid.shape[0] != 3:
            raise ValueError(f"Unexpected centroid shape: {centroid.shape}")
        cx3d, cy3d, cz3d = centroid


        median_distance = cz3d

        # Define 3D box corners in real-world coordinates
        corners3d = [
            [min_vals[0], min_vals[1], min_vals[2]],  # Front-bottom-left (0)
            [max_vals[0], min_vals[1], min_vals[2]],  # Front-bottom-right (1)
            [max_vals[0], max_vals[1], min_vals[2]],  # Front-top-right (2)
            [min_vals[0], max_vals[1], min_vals[2]],  # Front-top-left (3)
            [min_vals[0], min_vals[1], max_vals[2]],  # Back-bottom-left (4)
            [max_vals[0], min_vals[1], max_vals[2]],  # Back-bottom-right (5)
            [max_vals[0], max_vals[1], max_vals[2]],  # Back-top-right (6)
            [min_vals[0], max_vals[1], max_vals[2]]   # Back-top-left (7)
        ]

        # 3D Punkte zu 2D projizieren
        corners2d = [project_point(X, Y, Z, fx, fy, cx, cy) for (X, Y, Z) in corners3d]
        draw_3d_bounding_box(annotated_frame, corners2d, color)

        label = f"{class_name} {confidence:.2f}"
        draw_label_with_background(annotated_frame, label, x1, y1 - 10, color=(255, 255, 255), bg_color=color)
        print_object_info(class_name, median_distance, height_m, width_m, depth_m, confidence)

    return annotated_frame, object_colors


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    model = YOLO("yolov8n-seg.pt")
    object_colors = {}

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_maximum_distance = 15

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error: Could not open ZED camera.")
        exit()

    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy

    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path)

        depth_mat = sl.Mat()
        runtime_params = sl.RuntimeParameters()
        zed.grab(runtime_params)
        zed.retrieve_measure(depth_mat, sl.MEASURE.XYZ)  # Nutzung von XYZ statt DEPTH
        xyz_map = depth_mat.get_data()

        result, object_colors = process_frame(image, xyz_map, model, fx, fy, cx, cy, object_colors, device)
        cv2.imshow("YOLOv8 + ZED 3D Detection", result)
        while True:
            if cv2.getWindowProperty("YOLOv8 + ZED 3D Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        return

    else:
        image_mat = sl.Mat()
        depth_mat = sl.Mat()
        runtime_params = sl.RuntimeParameters()

        try:
            while True:
                if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image_mat, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth_mat, sl.MEASURE.XYZ)  #  Nutzung von XYZ
                    xyz_map = depth_mat.get_data()
                    frame = image_mat.get_data()
                    result, object_colors = process_frame(frame, xyz_map, model, fx, fy, cx, cy, object_colors, device)
                    
                    cv2.imshow("YOLOv8 + ZED 3D Detection", result)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        finally:
            cv2.destroyAllWindows()
            zed.close()


if __name__ == "__main__":
    main()