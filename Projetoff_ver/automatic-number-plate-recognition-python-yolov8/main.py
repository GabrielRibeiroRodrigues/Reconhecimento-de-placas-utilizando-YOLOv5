from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard  # Para detectar a tecla pressionada

from sort.sort import Sort
from util import get_car, read_license_plate, write_csv, check_plate_registration, load_registered_plates

results = {}
mot_tracker = Sort()

# Exemplo de uso na parte do seu código onde as placas são detectadas
registered_plates = load_registered_plates('C:\\Users\\Yasmin Pereira\\Desktop\\planilha.csv')

# Carregar os modelos
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO("C:\\Users\\Yasmin Pereira\\Desktop\\train3\\weights\\best.pt")

# Carregar vídeo
cap = cv2.VideoCapture("C:\\Users\\Yasmin Pereira\\Desktop\\ow.mp4")

vehicles = [2, 3, 5, 7]  # Definir as classes de veículos (ex: carro, caminhão, etc.)
min_confidence = 0.2  # Confiança mínima para detecção
frame_nmr = -1
ret = True

# Configurar matplotlib para exibição
plt.ion()  # Ativar modo interativo
fig, ax = plt.subplots()

# Leitura dos frames
while ret:
    frame_nmr += 1
    ret, frame = cap.read()  # Lê um frame do vídeo
    if not ret:  # Verifica se a leitura do frame foi bem-sucedida
        print("Não foi possível ler o frame. Encerrando.")
        break

    results[frame_nmr] = {}

    # Detecção de veículos usando o modelo de veículos
    detections_veiculos = coco_model(frame)[0]
    veiculos_detectados = []
    for detection in detections_veiculos.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
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
            # Verificação de limites
            if (0 <= x1 < frame.shape[1] and 0 <= x2 < frame.shape[1] and
                0 <= y1 < frame.shape[0] and 0 <= y2 < frame.shape[0]):
                # Recortar a placa para processamento
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Ler o texto da placa
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                print(f"Texto da placa detectado: {license_plate_text}, Confiança: {license_plate_text_score}")

                if license_plate_text is not None:
                    # Verificar se a placa já está registrada
                    if license_plate_text in registered_plates:
                        print(f"A placa {license_plate_text} já está registrada.")
                    else:
                        print(f"A placa {license_plate_text} não está registrada.")

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
                print(f"Coordenadas de recorte fora dos limites: ({x1}, {y1}), ({x2}, {y2})")
        else:
            print("Nenhum veículo correspondente à placa foi detectado.")

    # Exibir o frame atual
    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Converter BGR para RGB
    ax.axis('off')  # Ocultar os eixos

    # Desenhar bounding boxes para veículos detectados
    for veh in veiculos_detectados:
        x1, y1, x2, y2, score = veh
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='blue', linewidth=2)
        ax.add_patch(rect)

    # Desenhar bounding boxes para placas detectadas
    for plate in placas_detectadas:
        x1, y1, x2, y2, score = plate
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)

    plt.pause(0.001)  # Pausa para atualizar a visualização

    # Verificar se a tecla 'q' foi pressionada
    if keyboard.is_pressed('q'):
        print("Gravando dados e encerrando...")
        break  # Encerra o loop

# Escrever os resultados no CSV
print("Resultados finais antes de salvar no CSV:", results)
write_csv(results, 'C:\\Users\\Yasmin Pereira\\Desktop\\Projetoff-mat - droid\\test.csv')

# Fechar o vídeo
cap.release()
plt.close(fig)
