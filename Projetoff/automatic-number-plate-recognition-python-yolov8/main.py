from ultralytics import YOLO
import cv2
import numpy as np

from sort.sort import Sort
from util import get_car, read_license_plate, write_csv

results = {}
mot_tracker = Sort()

# Carregar os modelos
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO("C:\\Users\\suporte2\\Desktop\\train3\\weights\\best.pt")

# Carregar vídeo
cap = cv2.VideoCapture('C:\\Users\\suporte2\\Desktop\\video.mp4')

vehicles = [2, 3, 5, 7]  # Definir as classes de veículos (ex: carro, caminhão, etc.)
min_confidence = 0.2  # Confiança mínima para detecção
frame_nmr = -1
ret = True

# Leitura dos frames
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # if frame_nmr > 100:
        #       break
        results[frame_nmr] = {}

        # Detecção de veículos usando o modelo de veículos
        detections_veiculos = coco_model(frame)[0]
        veiculos_detectados = []
        for detection in detections_veiculos.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            print(f"Classe detectada: {class_id}, Score: {score}")
            if score >= min_confidence and int(class_id) in vehicles:
                veiculos_detectados.append([x1, y1, x2, y2, score])
        print(f"Frame {frame_nmr} - Veículos detectados: {veiculos_detectados}")

        # Rastrear veículos
        if veiculos_detectados:
            track_ids = mot_tracker.update(np.asarray(veiculos_detectados))
        else:
            track_ids = []
            print(f"Frame {frame_nmr} - Nenhum veículo detectado")

        # Detecção de placas usando o modelo de placas
        detections_placas = license_plate_detector(frame)[0]
        placas_detectadas = []
        for detection in detections_placas.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if score >= min_confidence:
                placas_detectadas.append([x1, y1, x2, y2, score])
        print(f"Frame {frame_nmr} - Placas detectadas: {placas_detectadas}")

        # Atribuir as placas aos veículos detectados
        for license_plate in placas_detectadas:
            x1, y1, x2, y2, score = license_plate
            print(f"Placa detectada no frame {frame_nmr} com coordenadas: ({x1}, {y1}), ({x2}, {y2}) e confiança {score}")

            # Verificar qual veículo corresponde à placa
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Recortar a placa para processamento
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Ler o texto da placa
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                print(f"Texto da placa detectado: {license_plate_text}, Confiança: {license_plate_text_score}")

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }
                else:
                    print("Nenhuma placa reconhecida.")
            else:
                print("Nenhum veículo correspondente à placa foi detectado.")

# Escrever os resultados no CSV
print("Resultados finais antes de salvar no CSV:", results)
write_csv(results, 'C:\\Users\\suporte2\\Desktop\\Projetoff\\test.csv')
