import cv2
import numpy as np
from models.base_model import BaseModel

class DilationModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.params = {
            'use_otsu': True,
            'manual_thresh': 127,
            'kernel_size': 5,
            'dilation_iters': 1,
        }
        
        self.display_modes = {
            0: "1. Imagem Original",
            1: "2. Tons de Cinza",
            2: "3. Limiarizada",
            3: "4. Mascara Dilatada",
            4: "5. Resultado Final"
        }

    def process(self, img):
        if img is None:
            return {}

        results = {}
        results['original'] = img.copy()
        
        gray = self.to_gray(img)
        results['gray'] = gray
        
        thresh = self.apply_threshold(
            gray, 
            use_otsu=self.params['use_otsu'], 
            manual_thresh=self.params['manual_thresh']
        )
        results['thresh'] = thresh.copy()
        
        dilated = self.apply_dilation(
            thresh, 
            kernel_size=self.params['kernel_size'], 
            iterations=self.params['dilation_iters']
        )
        results['dilated'] = dilated.copy()
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_contour = img.copy()
        cv2.drawContours(img_contour, contours, -1, (0, 0, 255), 2)
        
        for idx, c in enumerate(contours):
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cY, cX = np.mean(c, axis=0)[0].astype(int)
                
            cv2.circle(img_contour, (cX, cY), 3, (0, 255, 0), -1)
            cv2.putText(img_contour, str(idx + 1), (cX - 8, cY - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            
        results['segmented_overlay'] = img_contour
        results['count'] = len(contours)
        
        results['display_image_0'] = img
        results['display_image_1'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        results['display_image_2'] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        results['display_image_3'] = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        results['display_image_4'] = img_contour
        
        return results
