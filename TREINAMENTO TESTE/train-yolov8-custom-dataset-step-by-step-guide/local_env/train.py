from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="C:\\Users\\suporte2\\Desktop\\TREINAMENTO TESTE\\train-yolov8-custom-dataset-step-by-step-guide\\local_env\\config.yaml", epochs=10)  # train the model
