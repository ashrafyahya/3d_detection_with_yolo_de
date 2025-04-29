#In diesem Skript werden die Objekte zwar 3D erkennt, allerdings sind die Bounding Boxen 2D.
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

def main():
    # ZED-Kamera initialisieren
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_maximum_distance = 15  # Maximale Tiefe in Metern

    # Kamera öffnen
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Fehler: ZED-Kamera konnte nicht geöffnet werden.")
        exit()

    # Kamerakalibrierung abrufen
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    focal_length = calibration_params.left_cam.fx

    # YOLOv8-Modell laden
    model = YOLO("yolov8n.pt")  # Für Jetson Orin besser 'yolov8s.pt' oder 'yolov8m.pt' verwenden

    # Matrizen für ZED-Daten
    image = sl.Mat()
    depth = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    try:
        while True:
            # Bild und Tiefendaten erfassen
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT) #ein 2D-Bild von der linken Kamera.
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH) #eine Matrix, die die Tiefe (Entfernung) jedes Pixels im Bild enthält.

                # Bildkonvertierung
                frame = image.get_data()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

                # YOLOv8-Inferenz
                results = model(frame_rgb, verbose=False)
                annotated_frame = results[0].plot()

                # Für jede erkannte Bounding Box
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])

                    # 3D-Informationen extrahieren
                    depth_roi = depth.get_data()[y1:y2, x1:x2]#Schneidet einen Bereich (Region of Interest, ROI) aus der Tiefenmatrix aus.
                    valid_depths = depth_roi[np.isfinite(depth_roi)]
                    
                    if len(valid_depths) > 0:
                        # Distanzberechnung
                        avg_distance = np.mean(valid_depths)
                        
                        # Pixelgröße der Bounding Box
                        height_px = y2 - y1
                        width_px = x2 - x1
                        
                        # Reale Größe berechnen
                        height_m = (height_px * avg_distance) / focal_length
                        width_m = (width_px * avg_distance) / focal_length
                        
                        # Tiefe approximieren
                        depth_m = width_m * 0.7  # Anpassungsfaktor
                        
                        # Informationen ausgeben
                        print_object_info(
                            class_name=class_name,
                            distance=avg_distance,
                            height=height_m,
                            width=width_m,
                            depth=depth_m,
                            confidence=confidence
                        )

                # Bild anzeigen
                cv2.imshow("YOLOv8 + ZED 2D Detection", annotated_frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    finally:
        # Aufräumen
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()