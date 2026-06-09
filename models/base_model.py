import cv2
import numpy as np

class BaseModel:

    def __init__(self):
        self.params = {}
        self.display_modes = {}

    def set_param(self, name, value):
        """Define o valor de um parâmetro, se ele existir no modelo."""
        if name in self.params:
            self.params[name] = value

    def get_param(self, name):
        """Retorna o valor de um parâmetro do modelo."""
        return self.params.get(name)

    def process(self, img):
        raise NotImplementedError("Cada modelo de segmentação deve implementar o método process.")

    def to_gray(self, img):
        """Converte uma imagem colorida BGR em tons de cinza."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    def apply_smoothing(self, img, filter_type, blur_size, pyr_spatial_rad=15, pyr_color_rad=30):
        """
        Aplica filtro de suavização à imagem BGR:
        - 0: Nenhum
        - 1: Gaussian Blur
        - 2: pyrMeanShiftFiltering (suavização que preserva bordas)
        """
        if filter_type == 1:
            ksize = blur_size
            if ksize % 2 == 0:
                ksize += 1
            return cv2.GaussianBlur(img, (ksize, ksize), 0)
        elif filter_type == 2:
            return cv2.pyrMeanShiftFiltering(img, pyr_spatial_rad, pyr_color_rad)
        return img.copy()

    def apply_threshold(self, gray, use_otsu=True, manual_thresh=127, invert=True):
        """
        Aplica limiarização simples à imagem em tons de cinza.
        """
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        if use_otsu:
            _, thresh = cv2.threshold(gray, 0, 255, thresh_type | cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, manual_thresh, 255, thresh_type)
        return thresh

    def apply_adaptive_threshold(self, gray, block_size=51, c_val=30):
        """
        Aplica limiarização adaptativa (Mean C).
        c_val representa o valor cru vindo do slider (devemos mapear subtraindo 50 na chamada se necessário).
        """
        bs = block_size
        if bs % 2 == 0:
            bs += 1
        if bs < 3:
            bs = 3
            
        c_actual = float(c_val - 50)
        return cv2.adaptiveThreshold(
            gray, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, bs, c_actual
        )

    def apply_erosion(self, binary, kernel_size, iterations=1):
        """Aplica erosão morfológica em uma máscara binária."""
        if kernel_size <= 0 or iterations <= 0:
            return binary.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.erode(binary, kernel, iterations=iterations)

    def apply_dilation(self, binary, kernel_size, iterations=1):
        """Aplica dilatação morfológica em uma máscara binária."""
        if kernel_size <= 0 or iterations <= 0:
            return binary.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.dilate(binary, kernel, iterations=iterations)

    def apply_hole_filling(self, binary, max_area=1000):
        """Preenche buracos escuros internos em uma máscara branca."""
        if max_area <= 0:
            return binary.copy()
            
        filled = binary.copy()
        contours, _ = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < max_area:
                holes.append(c)
        if holes:
            cv2.drawContours(filled, holes, -1, 255, -1)
        return filled
