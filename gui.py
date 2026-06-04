import cv2
import numpy as np

class ViewerGUI:
    """
    Classe responsável pela Interface Gráfica usando OpenCV.
    Gerencia as janelas, desenho de overlay semitransparente,
    redimensionamento e sliders (trackbars) para sintonia de parâmetros de cada modelo.
    """
    def __init__(self, main_window_name="Visualizador de Feijoes", control_window_name="Controle de Parametros"):
        self.main_window = main_window_name
        self.control_window = control_window_name
        self.show_overlay = True
        
    def create_windows(self):
        """Cria e configura as janelas do OpenCV."""
        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.control_window, 600, 520)
        
        header = np.zeros((60, 600, 3), dtype=np.uint8)
        header[:] = (30, 30, 30)
        cv2.putText(header, "Painel de Controle", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(header, "Ajuste os valores para sintonizar a contagem", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow(self.control_window, header)

    def _dummy_callback(self, val):
        """Callback vazio obrigatório para a criação de trackbars."""
        pass

    def setup_trackbars(self, model, model_idx=0):
        """
        Cria os controles deslizantes específicos para o modelo de segmentação ativo.
        Limpa controles antigos destruindo e recriando a janela.
        """
        try:
            cv2.destroyWindow(self.control_window)
        except cv2.error:
            pass
            
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.control_window, 600, 520)
        
        header = np.zeros((60, 600, 3), dtype=np.uint8)
        header[:] = (30, 30, 30)
        cv2.putText(header, "Painel de Controle", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(header, "Ajuste os valores para sintonizar a contagem", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow(self.control_window, header)
        
        cv2.createTrackbar("Modelo", self.control_window, model_idx, 2, self._dummy_callback)
        
        max_mode = len(model.display_modes) - 1
        cv2.createTrackbar("Exibir", self.control_window, max_mode, max_mode, self._dummy_callback)
        
        from models.watershed_model import WatershedModel
        from models.erosion_model import ErosionModel
        from models.dilation_model import DilationModel
        
        if isinstance(model, WatershedModel):
            cv2.createTrackbar("Mascara", self.control_window, model.params['mask_source'], 2, self._dummy_callback)
            cv2.createTrackbar("Filtro", self.control_window, model.params['filter_type'], 2, self._dummy_callback)
            
            initial_blur = (model.params['blur_size'] - 1) // 2
            cv2.createTrackbar("Blur", self.control_window, initial_blur, 15, self._dummy_callback)
            
            cv2.createTrackbar("Otsu", self.control_window, 1 if model.params['use_otsu'] else 0, 1, self._dummy_callback)
            cv2.createTrackbar("Limiar", self.control_window, model.params['manual_thresh'], 255, self._dummy_callback)
            
            initial_fill = model.params['fill_holes_area'] // 50
            cv2.createTrackbar("Buracos", self.control_window, initial_fill, 100, self._dummy_callback)
            
            cv2.createTrackbar("Erosao Sz", self.control_window, model.params['erosion_size'], 15, self._dummy_callback)
            cv2.createTrackbar("Erosao It", self.control_window, model.params['erosion_iters'], 10, self._dummy_callback)
            cv2.createTrackbar("Dilat Sz", self.control_window, model.params['dilation_size'], 15, self._dummy_callback)
            cv2.createTrackbar("Dilat It", self.control_window, model.params['dilation_iters'], 10, self._dummy_callback)
            cv2.createTrackbar("Dist Min", self.control_window, model.params['min_distance'], 100, self._dummy_callback)
            
        elif isinstance(model, ErosionModel):
            cv2.createTrackbar("Adapt.", self.control_window, 1 if model.params['use_adaptive'] else 0, 1, self._dummy_callback)
            
            initial_block = (model.params['block_size'] - 3) // 2
            cv2.createTrackbar("Bloco", self.control_window, initial_block, 49, self._dummy_callback)
            
            cv2.createTrackbar("Const. C", self.control_window, model.params['c_val'], 100, self._dummy_callback)
            
            cv2.createTrackbar("Otsu", self.control_window, 1 if model.params['use_otsu'] else 0, 1, self._dummy_callback)
            cv2.createTrackbar("Limiar", self.control_window, model.params['manual_thresh'], 255, self._dummy_callback)
            
            cv2.createTrackbar("Erosao Sz", self.control_window, model.params['kernel_size'], 15, self._dummy_callback)
            cv2.createTrackbar("Erosao It", self.control_window, model.params['erosion_iters'], 10, self._dummy_callback)
            
        elif isinstance(model, DilationModel):
            cv2.createTrackbar("Otsu", self.control_window, 1 if model.params['use_otsu'] else 0, 1, self._dummy_callback)
            cv2.createTrackbar("Limiar", self.control_window, model.params['manual_thresh'], 255, self._dummy_callback)
            
            cv2.createTrackbar("Dilat Sz", self.control_window, model.params['kernel_size'], 15, self._dummy_callback)
            cv2.createTrackbar("Dilat It", self.control_window, model.params['dilation_iters'], 10, self._dummy_callback)

    def update_params_from_trackbars(self, model):
        """
        Lê a posição dos sliders e atualiza os parâmetros do modelo ativo.
        Retorna uma tupla contendo (modelo_selecionado_idx, modo_exibicao_ativo).
        """
        selected_model_idx = cv2.getTrackbarPos("Modelo", self.control_window)
        display_mode = cv2.getTrackbarPos("Exibir", self.control_window)
        
        if selected_model_idx < 0: selected_model_idx = 0
        if display_mode < 0: display_mode = 0
        
        from models.watershed_model import WatershedModel
        from models.erosion_model import ErosionModel
        from models.dilation_model import DilationModel
        
        if isinstance(model, WatershedModel):
            model.set_param('mask_source', cv2.getTrackbarPos("Mascara", self.control_window))
            model.set_param('filter_type', cv2.getTrackbarPos("Filtro", self.control_window))
            
            blur_slider = cv2.getTrackbarPos("Blur", self.control_window)
            model.set_param('blur_size', (blur_slider * 2) + 1)
            
            use_otsu = cv2.getTrackbarPos("Otsu", self.control_window) == 1
            model.set_param('use_otsu', use_otsu)
            model.set_param('manual_thresh', cv2.getTrackbarPos("Limiar", self.control_window))
            
            fill_slider = cv2.getTrackbarPos("Buracos", self.control_window)
            model.set_param('fill_holes_area', fill_slider * 50)
            
            model.set_param('erosion_size', cv2.getTrackbarPos("Erosao Sz", self.control_window))
            model.set_param('erosion_iters', max(1, cv2.getTrackbarPos("Erosao It", self.control_window)))
            model.set_param('dilation_size', cv2.getTrackbarPos("Dilat Sz", self.control_window))
            model.set_param('dilation_iters', max(1, cv2.getTrackbarPos("Dilat It", self.control_window)))
            model.set_param('min_distance', cv2.getTrackbarPos("Dist Min", self.control_window))
            
        elif isinstance(model, ErosionModel):
            use_adaptive = cv2.getTrackbarPos("Adapt.", self.control_window) == 1
            model.set_param('use_adaptive', use_adaptive)
            
            block_slider = cv2.getTrackbarPos("Bloco", self.control_window)
            model.set_param('block_size', (block_slider * 2) + 3)
            
            model.set_param('c_val', cv2.getTrackbarPos("Const. C", self.control_window))
            
            use_otsu = cv2.getTrackbarPos("Otsu", self.control_window) == 1
            model.set_param('use_otsu', use_otsu)
            model.set_param('manual_thresh', cv2.getTrackbarPos("Limiar", self.control_window))
            
            model.set_param('kernel_size', cv2.getTrackbarPos("Erosao Sz", self.control_window))
            model.set_param('erosion_iters', max(1, cv2.getTrackbarPos("Erosao It", self.control_window)))
            
        elif isinstance(model, DilationModel):
            use_otsu = cv2.getTrackbarPos("Otsu", self.control_window) == 1
            model.set_param('use_otsu', use_otsu)
            model.set_param('manual_thresh', cv2.getTrackbarPos("Limiar", self.control_window))
            
            model.set_param('kernel_size', cv2.getTrackbarPos("Dilat Sz", self.control_window))
            model.set_param('dilation_iters', max(1, cv2.getTrackbarPos("Dilat It", self.control_window)))
            
        return selected_model_idx, display_mode

    def resize_to_fit(self, image, max_width=1280, max_height=720):
        """Redimensiona proporcionalmente a imagem para caber nos limites definidos."""
        h, w = image.shape[:2]
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
        return image, 1.0

    def draw_overlay(self, image, filename, idx, total, bean_count, current_mode_str, active_model_name):
        """Desenha uma barra informativa estilizada na parte inferior da imagem."""
        if not self.show_overlay:
            return image
            
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        banner_height = 50
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (0, h - banner_height), (w, h), (15, 15, 15), -1)
        
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.43
        text_color = (245, 245, 245)
        thickness = 1
        
        info_text = f"[{idx+1}/{total}] {filename}  |  Modelo: {active_model_name}  |  Total Detectado: {bean_count}"
        cv2.putText(img_copy, info_text, (15, h - 30), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        mode_text = f"Exibicao: {current_mode_str}"
        cv2.putText(img_copy, mode_text, (15, h - 12), font, font_scale, (100, 200, 255), thickness, cv2.LINE_AA)
        
        help_text = "Teclas: A/D/Setas (Navegar) | 1:Watershed | 2:Erosao | 3:Dilatacao | H (Menu) | ESC (Sair)"
        text_size = cv2.getTextSize(help_text, font, font_scale, thickness)[0]
        cv2.putText(img_copy, help_text, (w - text_size[0] - 15, h - 20), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        return img_copy

    def show(self, image, title_info=""):
        """Exibe a imagem na janela principal."""
        title = f"{self.main_window} {title_info}"
        cv2.setWindowTitle(self.main_window, title)
        cv2.imshow(self.main_window, image)

    def close(self):
        """Fecha todas as janelas gerenciadas pelo OpenCV."""
        cv2.destroyAllWindows()
