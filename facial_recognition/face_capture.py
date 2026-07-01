import cv2
import os
from datetime import datetime
from picamera2 import Picamera2
import time

PERSON_NAME = input('Введите имя: ')

def create_folder(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

def capture_photos(name):
    folder = create_folder(name)
    
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()
    time.sleep(2)

    photo_count = 0
    
    print(f"Создание фоторографий для {name}. Нажмите ПРОБЕЛ чтобы сделать снимок, 'q' для завершения.")
    
    while True:
        frame = picam2.capture_array()
        cv2.imshow('Capture', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Фоторография №{photo_count} сохранена: {filepath}")
        
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()
    print(f"Сохранение фоторографий завершено. {photo_count} фоторографий под именем {name}.")

if __name__ == "__main__":
    capture_photos(PERSON_NAME)
