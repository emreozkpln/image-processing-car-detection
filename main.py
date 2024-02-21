import cv2 #  Videodan karelerin okunması, sınırlayıcı kutuların çizimi, görüntülerin ekranda gösterilmesi gibi temel görüntü işleme görevleri için kullanılır.
import math #  Güven değerini daha anlaşılır bir formatta ifade etmek için kullanılır.
from ultralytics import YOLO
import cvzone # Sınırlayıcı kutuların köşelerinin çizimi için kullanılır.
import numpy as np

video_path = "C:\\Users\\emreo\\OneDrive\\Masaüstü\\herşey\\car.mp4"
cap = cv2.VideoCapture(video_path)

class_names =[ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

# Daha hızlı bir işlem için her 2 frame'i işle
# Frame videodaki tek bir görüntü karesini temsil eder
frame_skip = 2
frame_count = 0

# Hazır model
model = YOLO("yolov8l.pt")

while True:
    # Kameradan bir kare oku
    success, img = cap.read()
    if success==False:
        break

    frame_count += 1

    # Her iki frameden birini işle
    if frame_count % frame_skip == 0:
        # YOLO modelini kullanarak nesne tespiti yap
        # Sürekli gelen görüntülerden dolayı dizi işlemleri için stream=True kullanılır
        results = model(img, stream=True)

        for i in results:
            # Her bir sonuç için sınırlayıcı kutuları al
            boxes = i.boxes

            for box in boxes:
                # Güven değerini ve sınıf etiketini al
                # box.conf güven değerini alır 0la 1 arasında
                # math.ceil güven değerini yüzde cinsinden alır
                conf = math.ceil((box.conf[0] * 100)) / 100
                # box.cls sınıfı temsil eder. Örneğin araba için box.cls araba değerinin indexsini belirler.
                cls = int(box.cls[0])

                # Sadece "car", "bus", ve "truck" sınıflarını işle
                if class_names[cls] in ["car", "bus", "truck"]:
                    # Sınırlayıcı kutuyu al ve koordinatları ayarla
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Sınırlayıcı kutunun genişliği ve yüksekliği
                    w, h = (x2 - x1), (y2 - y1)

                    # Eğer boyutlar 0 veya negatifse, bu durumu ele al
                    if w > 0 and h > 0:
                        # Sınırlayıcı kutuyu çiz
                        # Sırayla parametre ,kutunun köşe kordinatları , renk rgb, çizgi kalınlığı
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                        # Köşeleri çiz
                        # Kutunun sol üst köşesi genişlik ve yükseklik , köşe uzunluğu
                        cvzone.cornerRect(img, (x1, y1, w, h), l=15)

                        # Sınıf adını ve güven değerini metin olarak çiz
                        text = f"{class_names[cls]} {conf}"

                        # Etiketi sınırlayıcı kutunun üzerine yaz
                        # Resim, metin, metnin başlayacağı konumun koordinatları, yazı tipi, yazıtipi ölçeği, rengi, kalınlığı
                        cv2.putText(img, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Sonuçları göster
        cv2.imshow("Object Detection", img)

    # Klavyeden bir tuşa basıldığını kontrol et
    key = cv2.waitKey(1) & 0xFF

    # Eğer 'q' tuşuna basıldıysa döngüden çık
    if key == ord('q'):
        break

# Pencereyi kapat
cap.release()
# Açık olan tüm opencv pencerelerini kapaıtr
cv2.destroyAllWindows()
