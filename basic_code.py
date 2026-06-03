#####################################
## WATERSHED
#####################################
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import color

img = cv2.imread("coins.jpeg")

filtro = cv2.pyrMeanShiftFiltering(img, 20, 40)
gray = cv2.cvtColor(filtro, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Preenchimento de falhas nos objetos binarizados
contornos, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
falhas = []
for con in contornos:
    area = cv2.contourArea(con)
    if area < 1000:
        falhas.append(con)
cv2.drawContours(thresh, falhas, -1, 255, -1)
preenchida = thresh.copy() # copia para não perder dados e exibir thresh no final

# cálculo da distância da transformação
dist = cv2.distanceTransform(preenchida, distanceType=cv2.DIST_L2, maskSize=3, dstType=cv2.CV_8U)
dist = cv2.normalize(dist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

local_maxi = peak_local_max(dist, min_distance=20, labels=preenchida)
local_max = np.zeros_like(dist, dtype=bool)
local_max[tuple(local_maxi.T)] = True

# marcando os labels
markers = ndi.label(local_max, structure=np.ones((3,3)))[0]

#watershed
labels = watershed(-dist, markers, mask=thresh)
print("Objetos encontrados: ", len(np.unique(labels)) - 1)
imgWatershed = color.label2rgb(labels, bg_label=0)

cv2.imshow("1 Imagem original", img)
cv2.imshow("3 Imagem preenchida", thresh)
cv2.imshow("4 Calculo Dist.", dist)
cv2.imshow("5 Watershed", imgWatershed)

cv2.waitKey(0)
cv2.destroyAllWindows()


###################################
# EROSAO
###################################
# import cv2 as cv
# import numpy as np
#
# imgRice = cv.imread("rice.png", 0)
#
# imgBinary = cv.adaptiveThreshold (imgRice, 255.0,
#     cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, -20.0)
# cv.imshow("Imagem Binarizada", imgBinary)
#
# # Aplicação de erosão
# kernel = np.ones((5,5),np.uint8)
# imgErosion = cv.erode(imgBinary, kernel)
# cv.imshow("Imagem com Erosão", imgErosion)
#
# contours, _ = cv.findContours(imgErosion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# imgContour = cv.cvtColor(imgRice, cv.COLOR_GRAY2BGR)
#
# cv.drawContours(imgContour, contours, -1, (0, 0, 255), 2)
#
# print("Total de objetos detectados: ", len(contours))
# cv.imshow("Objetos detectados", imgContour)
#
# cv.waitKey(0)
# cv.destroyAllWindows()


################################################
# DILATACAO
################################################
# import cv2 as cv
# import numpy as np
#
# imgRice = cv.imread("coins.jpeg", 0)
#
# _,imgBinary = cv.threshold(imgRice,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# cv.imshow("Imagem Binarizada", imgBinary)
#
# ## Dilatamento da imagem para reduzir ruído
# kernel = np.ones((5,5),np.uint8)
# imgDilated = cv.dilate(imgBinary, kernel)
# cv.imshow("Imagem dilatada", imgDilated)
#
# contours, _ = cv.findContours(imgDilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# imgContour = cv.cvtColor(imgRice, cv.COLOR_GRAY2BGR)
#
# cv.drawContours(imgContour, contours, -1, (0, 0, 255), 2)
#
# print("Total de objetos detectados: ", len(contours))
# cv.imshow("Objetos detectados", imgContour)
#
# cv.waitKey(0)
# cv.destroyAllWindows()