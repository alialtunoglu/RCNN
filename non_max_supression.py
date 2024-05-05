import numpy as np
import cv2

#Maksimum olmayan bastırma

#Örneğin bir yüzü birden fazla kare ile tespit ettik ama bize 1 tanesi lazım tüm karelerin kesişimine bakıp tüm alana böleriz
#ve bu belli bir eşik değerin üstündeyse onu alırız o eşik değere overlapThresh diyoruz

#probs -> Bizim çıkan sonuçlarımız -> Yani 3 tane penceremiz varsa bu 3 tane penceremiz içerisindeki en 
#yüksek sınıflandırma derecesine sahip olan pencereyi almak daha iyidir.


def non_max_suppression(boxes, probs = None, overlapThresh=0.3):
    
    if len(boxes) == 0:
        return [] #Eğer gelen kutular boşda boş bir liste döndürüyoruz
    
    #gelen kutularımızın datatype'ı integer ise float'a çeviriyoruz
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
        
    #x1,y1 kutunun başlangıç noktası
    #x2,y2 kutunun sonlanma noktası
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    
    
    # alanı bulalım -> arada 1 piksel kaybettiğimiz için ekliyoruz
    area = (x2 - x1 + 1)*(y2 - y1 + 1)
    
    #alt sol köşedeki y kordinatı
    idxs = y2
    
    
    # olasılık degerleri None değilse indexlerimizi olasılığa göre sıralayacağız
    if probs is not None:
        idxs = probs
        
        
    # indeksi sırala
    idxs = np.argsort(idxs)
    
    pick = [] # secilen kutular
    
    while len(idxs) > 0:
        
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # en buyuk ve en küçük x ve y
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # w,h bul
        w = np.maximum(0,xx2 - xx1 + 1)
        h = np.maximum(0,yy2 - yy1 + 1)
        
        # overlap 
        overlap = (w*h)/area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    return boxes[pick].astype("int")
        
    
    
    
    
    
    
    
    
    