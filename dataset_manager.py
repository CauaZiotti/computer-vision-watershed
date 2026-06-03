import os

class DatasetManager:
    def __init__(self, directory="dataset", extensions=None):
        self.directory = directory
        self.extensions = extensions or ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
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
