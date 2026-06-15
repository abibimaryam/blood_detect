from ultralytics import YOLO
import os

def train_model():
    # Путь к файлу конфигурации датасета
    # Используем абсолютный путь, чтобы избежать проблем с поиском файлов
    data_path = r'D:\projects\blood_detect\Robo dataset.v15i.yolov8\data.yaml'
    
    # Загружаем предобученную модель YOLOv8 (nano версия — самая быстрая)
    # Вы можете заменить на 'yolov8s.pt', 'yolov8m.pt' и т.д. для более высокой точности
    model = YOLO('yolov8m.pt')

    # Запуск обучения
    results = model.train(
        data=data_path,
        epochs=300,
        imgsz=640,
        batch=8,
        name='blood_detect_v1',
        device=0,
        patience=15  # Early Stopping после 15 эпох без улучшения
    )
    
    print("Обучение завершено. Результаты сохранены в папке 'runs/detect/blood_detect_v1'")

if __name__ == '__main__':
    train_model()
