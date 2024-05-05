import cv2
import matplotlib.pyplot as plt

#minSize -> burada bir eşik değeri olacak bu değere ulaşıldığında durmayı sağlar
def image_pyramid(image, scale = 1.5, minSize=(224,224)):
    
    yield image
    #yield bir jeneratördür
    
    while True:
        
        #image.shape[1] -> genişlik kısacası satır*sutun width->sutun height-> satır
        #ımage.shape[0] -> yükseklik
        w = int(image.shape[1]/scale)
        image = cv2.resize(image, dsize=(w,w))
        
        #resmi karesel bir şekilde yeniden boyutlandırıyoruz
        
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        yield image
      
'''
img = cv2.imread("husky.jpg")
im = image_pyramid(img,1.5, (10,10))
for i, image in enumerate(im):
    print(im)
    print(i)
    if i == 3:
        plt.imshow(image)
        
'''
'''
enumerate, bir dizi, liste veya diğer veri yapıları üzerinde birlikte döngü yaparken 
hem elemanın kendisini hem de o elemanın dizinini elde etmek için kullanılan bir Python 
işlevi veya fonksiyonudur.
'''