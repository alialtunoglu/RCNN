# 1. Kütüphaneleri Yükleme:
from tensorflow.keras.applications.resnet50 import preprocess_input 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

from sliding_window import sliding_window
from image_pyramid import image_pyramid
from non_max_supression import non_max_suppression


#2. Parametre Tanımlama:

# ilklendirme parametreleri
WIDTH = 600
HEIGHT = 600
PYR_SCALE = 1.5 # Görüntü piramidinde her bir seviye arasındaki küçültme oranı.
WIN_STEP = 16 #Kayan pencerenin 1 kerede kaç pixel kayacağı
ROI_SIZE = (200,150) #Kayan pencerenin boyutu
INPUT_SIZE = (224, 224) # ResNet50 modelinin giriş boyutu.



#3. ResNet Modelini Yükleme:

# Kod, önceden eğitilmiş ResNet50 derin öğrenme modelini "imagenet" ağırlıklarıyla yükler. 
#Bu model, 1000 farklı nesneyi sınıflandırmak için eğitilmiştir.
print("Resnet yukleniyor")
model = ResNet50(weights = "imagenet", include_top = True)


#4. Orijinal Görüntüyü Yükleme ve İşleme:

orig = cv2.imread("husky.jpg") # "husky.jpg" adlı görüntüyü okur.
orig = cv2.resize(orig, dsize = (WIDTH, HEIGHT)) # Görüntüyü WIDTH ve HEIGHT boyutlarına yeniden boyutlandırır.
cv2.imshow("Husky",orig)

(H,W) = orig.shape[:2] #600*600


#5. Görüntü Piramidi Oluşturma:
# image pyramid
pyramid = image_pyramid(orig, PYR_SCALE, ROI_SIZE)



#6. Kayan Pencere ile Bölge Çıkarma:

rois = []
locs = []
scales = []
i=0
# Oluşturulan görüntü piramidindeki her bir görüntü için döngü başlar.
for image in pyramid:
    
    # Her görüntü için küçültme oranı hesaplanır.
    # Küçültme oranı = (resmin normal genişliği)/(piramid ile küçültülmüş resmin genişliği) 
    scale = W/float(image.shape[1]) 
    scales.append(scale)
    for (x,y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        # Bu fonksiyon görüntü üzerinde kayan pencere oluşturarak WIN_STEP aralığında
        #ROI_SIZE boyutunda bölgeler çıkarır.
        #Buradaki x,y'ler dikdörtgenin köşe kordinatlarının başlangıcını tutuyor
        
        x = int(x*scale)
        y = int(y*scale)
        w = int(ROI_SIZE[0]*scale)
        h = int(ROI_SIZE[1]*scale)
        
        #Burada her çıkarılan bölge ResNet50 modeline girdi olarak veriliyor.
        #Çıkarılan bölge, ResNet50 modelinin giriş boyutu olan INPUT_SIZE'a yeniden boyutlandırılır.
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        #OpenCV görüntüsü Numpy dizisine dönüştürülür.
        roi = img_to_array(roi)
        #ResNet50 modelinin beklediği ön işleme adımları uygulanır.
        roi = preprocess_input(roi) 
        #Bu fonksiyon, girdi verilerini modelin beklentilerine uygun hale getirir ve 
        #genellikle veriyi standartlaştırır, normalleştirir veya ölçeklendirir.
     
        #İşlenen bölge, rois listesine eklenir.
        #Kayan pencerenin konumu (x, y, genişlik, yükseklik) bilgisi locs listesine eklenir.
        rois.append(roi)
        locs.append((x,y,x+w,y+h))
       

#7. Sınıflandırma:

rois = np.array(rois, dtype = "float32")
# Çıkarılan ve ön işlenen bölgeler, Numpy dizisine dönüştürülür.


# ResNet50 modeli, tüm bölgelere uygulanarak sınıflandırma sonuçları preds değişkenine atanır.
print("sınıflandırma işlemi")
preds = model.predict(rois)

# Yukarıdaki Bu kod blokları, rois adlı bir listeyi Numpy dizisine dönüştürüyor ve 
#sonra ResNet50 modeline bu bölgeleri girdi olarak veriyor. Sonuçlar, modelin sınıflandırma 
#tahminlerini içeren bir dizi olarak preds değişkenine atanıyor.


#8. Sonuçları Deşifre Etme:
'''
Bu kod, derin öğrenme modeli tarafından yapılan sınıflandırma tahminlerini insan tarafından okunabilir 
bir forma dönüştürmek için kullanılır. Özellikle, genellikle ImageNet veri kümesindeki etiketlerle 
eşleştirilen tahminleri döndürür.
''' 
preds = imagenet_utils.decode_predictions(preds, top = 1)



#9. Nesne Tespiti ve Çizim:

#Tespit edilen nesnelerin etiketlerini ve konumlarını tutmak için boş bir sözlük oluşturulur.
labels = {}
#Minimum güven skoru belirlenir. Sadece bu skordan yüksek olasılıkla tespit edilen nesneler işleme alınır.
min_conf = 0.9

#Her bir bölgenin sınıflandırma sonucu için döngü başlar.
for (i,p) in enumerate(preds):
    
    (_, label, prob) = p[0] # ('n02109961', 'Eskimo_dog', 0.46597967) -> -,label, prob
    
    #Yüzde 90 olasılıktan büyük olanları sadece hesaplıyorum
    if prob >= min_conf:
        
        box = locs[i] #Kayan pencerenin konumu alınır.
        
        #fonksiyonu, label anahtarına karşılık gelen değeri bulamazsa, boş bir liste ([]) döndürür.
        L = labels.get(label, []) 
        L.append((box, prob))
        #((120, 264, 420, 489), 0.90466434) box-> nesnenin konum bilgisini içeren tuple , 
        #prob -> nesnenin sınıflandırılma olasılığını temsil eden bir sayıdır.

        labels[label] = L


# Tespit edilen tüm nesne etiketleri için döngü başlar.
for label in labels.keys(): 
    
    #print(label) # Eskimo_dog geliyor
    
    
    # Orijinal görüntünün bir kopyası oluşturulur.
    clone = orig.copy()
    
    #Belirli bir etiket için tespit edilen tüm nesneler için döngü başlar.
    #Eskimo_dog etiketinin tespit edildiği 7 indis var 1.si -> ((120, 264, 420, 489), 0.90466434)
    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box # Tespit edilen yerlerin başlangıç ve bitiş yerlerini değişkene atıyor
        #attığı o değişkeni kullanarak clone resmin tespit edilen bölgesine 1 tane kare çiziyor
        cv2.rectangle(clone, (startX, startY),(endX, endY), (0,255,0),2) 
        
    
    cv2.imshow("ilk",clone)
    
    
    clone = orig.copy()
    
    # non-maxima
    # Tespit edilen nesnelerin konum bilgileri bir Numpy dizisine dönüştürülür.
    boxes = np.array([p[0] for p in labels[label]])
    #Tespit edilen nesnelerin olasılıkları bir Numpy dizisine dönüştürülür.
    proba = np.array([p[1] for p in labels[label]])
    
    
    
    boxes = non_max_suppression(boxes, proba)
    #fonksiyonu kullanılarak üst üste binen nesnelerden sadece olasılığı en yüksek olanı bırakılır.
    
    
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, startY),(endX, endY), (0,255,0),2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        #Bu satır, y değişkenine etiket metninin görüntünün hangi dikey konumuna yerleştirileceğini belirler.
        #Eğer startY 10'dan büyükse, y değeri startY - 10 olarak hesaplanır. Bu, etiket metninin nesnenin üzerinde görüntülenmesini sağlar.
        #Eğer startY 10'dan küçükse veya eşit ise, y değeri startY + 10 olarak hesaplanır. Bu, etiket metninin nesnenin altında görüntülenmesini sağlar.
    
    
        cv2.putText(clone, label, (startX , y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,255,0),2)
        
    cv2.imshow("Maxima", clone)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break
    
