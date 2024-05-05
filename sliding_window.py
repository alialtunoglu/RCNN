import cv2
import matplotlib.pyplot as plt

#Kayan pencereler

#step -> Dikdörtgenin resim üzerinde dolaşırken kaç pixel kayacağağının sayısıdır.
#ws -> Dikdörtgenin boyutudur
def sliding_window(image, step, ws):
    height, width = image.shape[:2]  # Görüntünün yüksekliğini ve genişliğini alın yükseklik*genişlik
    
    for y in range(0, height - ws[1], step):
        for x in range(0, width - ws[0], step):
            yield (x, y, image[y:y+ws[1], x:x+ws[0]])

'''
img = cv2.imread("husky.jpg")
im = sliding_window(img, 50, (200, 150)) #genişlik*yükseklik

for i, image in enumerate(im):
    
    if i == 1:
        print("Başlangıç y kordinatı:", image[1], "Başlangıç x kordinatı:", image[0])
        plt.imshow(image[2])

#Buradaki x,y'ler dikdörtgenin köşe kordinatlarının başlangıcını tutuyor
'''