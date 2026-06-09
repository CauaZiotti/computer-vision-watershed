import cv2
import numpy as np
# pyrefly: ignore [missing-import]
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import color
from models.base_model import BaseModel

class WatershedModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.params = {
            'mask_source': 0,
            'filter_type': 1,
            'blur_size': 9,
            'pyr_spatial_rad': 15,
            'pyr_color_rad': 30,
            'use_otsu': True,
            'manual_thresh': 120,
            'fill_holes_area': 1000,
            'erosion_size': 0,
            'erosion_iters': 1,
            'dilation_size': 0,
            'dilation_iters': 1,
            'min_distance': 11,
        }
        
        self.display_modes = {
            0: "1. Imagem Original",
            1: "2. Suavizada/Filtrada",
            2: "3. Mascara de Entrada",
            3: "4. Transformada de Distancia",
            4: "5. Rotulos Watershed",
            5: "6. Resultado Final"
        }

    def process(self, img):
        if img is None:
            return {}

        results = {}
        results['original'] = img.copy()
        
        filter_type = self.params['filter_type']
        smoothed = self.apply_smoothing(
            img, 
            filter_type, 
            self.params['blur_size'], 
            self.params['pyr_spatial_rad'], 
            self.params['pyr_color_rad']
        )
        results['filtered'] = smoothed
        
        gray = self.to_gray(smoothed)
        results['gray'] = gray
        
        mask_source = self.params['mask_source']
        
        if mask_source == 1:
            from models.erosion_model import ErosionModel
            erosion_helper = ErosionModel()
            
            erosion_helper.set_param('use_adaptive', False)
            erosion_helper.set_param('use_otsu', self.params['use_otsu'])
            erosion_helper.set_param('manual_thresh', self.params['manual_thresh'])
            erosion_helper.set_param('kernel_size', self.params['erosion_size'])
            erosion_helper.set_param('erosion_iters', self.params['erosion_iters'])
            
            erosion_results = erosion_helper.process(smoothed)
            morphed = erosion_results['eroded']
            thresh = erosion_results['thresh']
            
        elif mask_source == 2:
            from models.dilation_model import DilationModel
            dilation_helper = DilationModel()
            
            dilation_helper.set_param('use_otsu', self.params['use_otsu'])
            dilation_helper.set_param('manual_thresh', self.params['manual_thresh'])
            dilation_helper.set_param('kernel_size', self.params['dilation_size'])
            dilation_helper.set_param('dilation_iters', self.params['dilation_iters'])
            
            dilation_results = dilation_helper.process(smoothed)
            morphed = dilation_results['dilated']
            thresh = dilation_results['thresh']
            
        else:
            thresh = self.apply_threshold(
                gray, 
                use_otsu=self.params['use_otsu'], 
                manual_thresh=self.params['manual_thresh']
            )
            filled = self.apply_hole_filling(thresh, self.params['fill_holes_area'])
            morphed = self.apply_erosion(filled, self.params['erosion_size'], self.params['erosion_iters'])
            morphed = self.apply_dilation(morphed, self.params['dilation_size'], self.params['dilation_iters'])
            
        results['thresh_raw'] = thresh.copy()
        results['morphed'] = morphed
        
        dist = cv2.distanceTransform(morphed, distanceType=cv2.DIST_L2, maskSize=5)
        dist_norm = cv2.normalize(dist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        dist_smooth = cv2.GaussianBlur(dist_norm, (5, 5), 0)
        
        results['dist_transform'] = dist_smooth
        
        min_dist = self.params['min_distance']
        if min_dist < 1:
            min_dist = 1
            
        local_maxi = peak_local_max(dist_smooth, min_distance=min_dist, labels=morphed)
        
        local_max = np.zeros_like(dist_smooth, dtype=bool)
        if len(local_maxi) > 0:
            local_max[tuple(local_maxi.T)] = True
            
        markers = ndi.label(local_max, structure=np.ones((3,3)))[0]
        
        labels = watershed(-dist, markers, mask=morphed)
        
        unique_labels = np.unique(labels)
        bean_count = len(unique_labels) - 1
        results['count'] = bean_count
        
        if bean_count > 0:
            img_ws_float = color.label2rgb(labels, bg_label=0)
            img_ws = np.uint8(img_ws_float * 255)
            img_ws = cv2.cvtColor(img_ws, cv2.COLOR_RGB2BGR)
        else:
            img_ws = np.zeros_like(img)
            
        results['watershed_labels'] = img_ws
        
        overlay_outlines = img.copy()
        for label_val in unique_labels:
            if label_val == 0:
                continue
                
            bean_mask = np.uint8(labels == label_val) * 255
            contours_bean, _ = cv2.findContours(bean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours_bean:
                cv2.drawContours(overlay_outlines, contours_bean, -1, (0, 255, 0), 2)
                
                M = cv2.moments(contours_bean[0])
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    coords = np.argwhere(labels == label_val)
                    cY, cX = np.mean(coords, axis=0).astype(int)
                    
                cv2.circle(overlay_outlines, (cX, cY), 3, (0, 0, 255), -1)
                            
        results['segmented_overlay'] = overlay_outlines
        
        results['display_image_0'] = img
        results['display_image_1'] = smoothed
        results['display_image_2'] = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
        results['display_image_3'] = dist_norm
        results['display_image_4'] = img_ws
        results['display_image_5'] = overlay_outlines
        
        return results
