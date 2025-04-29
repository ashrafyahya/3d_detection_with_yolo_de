# Dieses Skript unterscheidet sich vom 3d_detection_3d_bbox_m2.py mit durch die Tiefenrechnung.
# Hierbei wird der Schätzungfaktor 0.7 verwendet.

import os
import sys
import cv2
import torch
import random
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO


# Funktion zur Ausgabe von Objektinformationen
def print_object_info(class_name, distance, height, width, depth, confidence):
    print("\n" + "-" * 40)
    print(f"Object name: {class_name}") 
    print(f"Distance: {distance:.2f}m") 
    print(f"Height: {height:.2f}m") 
    print(f"Width: {width:.2f}m")
    print(f"Depth: {depth:.2f}m")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 40 + "\n") 

# Funktion zur Generierung einer zufälligen Farbe
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 

# Funktion zum Zeichnen eines 3D-Bounding-Box
def draw_3d_bounding_box(image, points2d, color):
    #    4-------5
    #   /|      /|
    #  / |     / |
    # 0-------1  |
    # |  |    |  |
    # |  7----|--6
    # | /     | /
    # 3-------2
    # Definition der 6 Flächen eines Würfels
    faces = [
        [0, 1, 2, 3],  # Vorderseite
        [4, 5, 6, 7],  # Rückseite
        [0, 1, 5, 4],  # Oberseite
        [2, 3, 7, 6],  # Unterseite
        [0, 3, 7, 4],  # Linke Seite
        [1, 2, 6, 5]   # Rechte Seite
    ]
    
    # Erstellen einer transparenten Overlay-Ebene
    overlay = image.copy()
    
    # Zeichnen jeder Fläche mit Transparenz
    for face in faces:
        try:
            pts = np.array([points2d[i] for i in face], np.int32) 
            pts = pts.reshape((-1, 1, 2)) 
            cv2.fillPoly(overlay, [pts], color)
        except Exception as e:
            print(e)
    
    # Überblenden des Overlays mit dem Originalbild
    alpha = 0.2  # Transparenzfaktor (0.0 - 1.0)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Zeichnen der Kanten der Box
    for i in range(4):
        # Vorderseiten-Kanten
        try:
            cv2.line(image, points2d[i], points2d[(i + 1) % 4], color, 2)
            # Rückseiten-Kanten
            from_ = i + 4
            to_ = (i + 1) % 4 + 4
            if from_ != 7 and to_ != 7:
                cv2.line(image, points2d[from_], points2d[to_], color, 2)

            if i + 4 != 7:
                # Verbindungskanten
                cv2.line(image, points2d[i], points2d[i + 4], color, 2)
        except Exception as e:
            print(e)

# Funktion zur Projektion eines 3D-Punkts auf das 2D-Bild
def project_point(X, Y, Z, fx, fy, cx, cy):
    u = int((X * fx) / Z + cx) 
    v = int((Y * fy) / Z + cy) 
    return (u, v)

# Funktion zum Zeichnen eines Labels mit Hintergrund
def draw_label_with_background(image, text, x, y, color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX  # Schriftart
    scale = 0.5  # Schriftgröße
    thickness = 1  # Strichstärke
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)  # Größe des Textes berechnen
    cv2.rectangle(image, (x, y - h - 4), (x + w, y), bg_color, -1)  # Hintergrund zeichnen
    cv2.putText(image, text, (x, y - 2), font, scale, color, thickness, cv2.LINE_AA)  # Text zeichnen

# Hauptfunktion zur Verarbeitung eines Bildes/Frame
def process_frame(frame, depth_data, model, fx, fy, cx, cy, object_colors, device):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konvertierung von BGR zu RGB
    results = model(frame_rgb, verbose=False, device=device)
    annotated_frame = frame.copy() 
   
    masks = results[0].masks.data if results[0].masks is not None else None

    # Verarbeitung jeder erkannten Bounding-Box
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinaten der Box
        class_id = int(box.cls[0])  
        class_name = model.names[class_id]  
        confidence = float(box.conf[0])
        if (confidence < 0.6):
            continue

        if class_name not in object_colors:
            object_colors[class_name] = generate_random_color()
        color = object_colors[class_name]

        # Tiefendatenverarbeitung falls verfügbar
        if depth_data is not None:
            if masks is not None:
                # Verwendung der Segmentierungsmaske für Tiefenwerte
                mask_resized = cv2.resize(masks[i].cpu().numpy(), (depth_data.shape[1], depth_data.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                valid_depths = depth_data[mask_binary == 1]
                valid_depths = valid_depths[np.isfinite(valid_depths)]
            else:
                # Fallback auf ROI-basierte Tiefenberechnung
                depth_roi = depth_data[y1:y2, x1:x2]
                valid_depths = depth_roi[np.isfinite(depth_roi)]
        else:
            valid_depths = np.array([])

        # Wenn gültige Tiefendaten vorhanden sind
        if len(valid_depths) > 0:
            median_distance = np.median(valid_depths) 
            height_px = y2 - y1 
            width_px = x2 - x1 

            # Umrechnung in reale Maße
            height_m = (height_px * median_distance) / fx
            width_m = (width_px * median_distance) / fx
            depth_m = width_m * 0.7  # 0.7 Geschätzte Tiefe

            # Berechnung der 3D-Eckpunkte der Box
            corners3d = []
            for dx, dy, dz in [(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)]:
                X = ((x1 + dx * width_px) - cx) * median_distance / fx
                Y = ((y1 + dy * height_px) - cy) * median_distance / fy
                Z = median_distance - depth_m / 2 + dz * depth_m

                if  not np.isfinite(X) or not np.isfinite(Y) or not np.isfinite(Z):
                    continue
                corners3d.append((X, Y, Z))
                
            # Projektion der 3D-Punkte auf 2D-Bild
            corners2d = [project_point(X, Y, Z, fx, fy, cx, cy) for (X, Y, Z) in corners3d]
            draw_3d_bounding_box(annotated_frame, corners2d, color) 

            # Beschriftung hinzufügen
            label = f"{class_name} {confidence:.2f}"
            draw_label_with_background(annotated_frame, label, x1, y1 - 10, color=(255, 255, 255), bg_color=color)

            # Objektinformationen ausgeben
            print_object_info(class_name, median_distance, height_m, width_m, depth_m, confidence)

    return annotated_frame, object_colors 

# Hauptfunktion
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Kommandozeilenargument für Bildpfad prüfen
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    model = YOLO("yolov8n-seg.pt")  
    object_colors = {} 

    # ZED-Kamera initialisieren
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Auflösung setzen
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Tiefenmodus
    init_params.coordinate_units = sl.UNIT.METER  # Einheiten in Metern
    init_params.depth_maximum_distance = 15  # Maximale Tiefe

    # Kamera öffnen
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Fehler: ZED-Kamera konnte nicht geöffnet werden.")
        exit()

    # Kamerakalibrierungsparameter holen
    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam 
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy 

    # Falls ein Bildpfad angegeben wurde
    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path) 

        # Tiefendaten von der Kamera holen
        depth_mat = sl.Mat()
        runtime_params = sl.RuntimeParameters()
        zed.grab(runtime_params)
        zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth_map = depth_mat.get_data()

        # Bild verarbeiten
        result, object_colors = process_frame(image, depth_map, model, fx, fy, cx, cy, object_colors, device)
        cv2.imshow("YOLOv8 + ZED 3D Detection", result)  
        while True:
            if cv2.getWindowProperty("YOLOv8 + ZED 3D Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):  
                break

        cv2.destroyAllWindows()  
        return

    else:
        # Live-Video-Modus
        image_mat = sl.Mat()
        depth_mat = sl.Mat()
        runtime_params = sl.RuntimeParameters()

        try:
            while True:
                if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                    # Bild und Tiefendaten holen
                    zed.retrieve_image(image_mat, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                    frame = image_mat.get_data()
                    depth_data = depth_mat.get_data()

                    result, object_colors = process_frame(frame, depth_data, model, fx, fy, cx, cy, object_colors, device)
                    cv2.imshow("YOLOv8 + ZED 3D Detection", result) 

                    if cv2.waitKey(10) & 0xFF == ord('q'): 
                        break
        finally:
            cv2.destroyAllWindows()  
            zed.close() 

if __name__ == "__main__":
    main() 