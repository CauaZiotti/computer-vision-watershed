import os
import cv2

class DatasetManager:
    def __init__(self, directory="dataset", extensions=None, max_dim=1280):
        self.directory = directory
        self.extensions = extensions or ('.jpg', '.jpeg', '.png')
        self.max_dim = max_dim
        self.images = []
        self._current_idx = 0
        self.load_dataset()
        
    def load_dataset(self):
        if not os.path.exists(self.directory):
            self.images = []
            return
            
        self.images = sorted([
            f for f in os.listdir(self.directory) 
            if f.lower().endswith(self.extensions)
        ])
        self._current_idx = 0
        
    @property
    def has_images(self):
        return len(self.images) > 0
        
    @property
    def total_images(self):
        return len(self.images)
        
    @property
    def current_index(self):
        return self._current_idx
        
    def get_current_filename(self):
        if not self.has_images:
            return None
        return self.images[self._current_idx]
        
    def get_current_image_path(self):
        filename = self.get_current_filename()
        if not filename:
            return None
        return os.path.join(self.directory, filename)
        
    def get_current_image(self):
        filepath = self.get_current_image_path()
        if not filepath:
            return None

        img = cv2.imread(filepath)
        if img is None:
            return None
            
        if self.max_dim:
            h, w = img.shape[:2]
            max_size = max(h, w)
            if max_size > self.max_dim:
                scale = self.max_dim / max_size
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
        
    def next_image(self):
        if not self.has_images:
            return None
        self._current_idx = (self._current_idx + 1) % len(self.images)
        return self.get_current_filename()
        
    def prev_image(self):
        if not self.has_images:
            return None
        self._current_idx = (self._current_idx - 1) % len(self.images)
        return self.get_current_filename()
