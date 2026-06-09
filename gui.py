import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

class ViewerGUI:
    """
    Classe responsável pela Interface Gráfica usando OpenCV e Tkinter.
    Gerencia as janelas, desenho de overlay semitransparente,
    redimensionamento proporcional e controles via Tkinter para sintonia de parâmetros.
    """
    def __init__(self, main_window_name="Visualizador de Feijoes", control_window_name="Controle de Parametros"):
        self.main_window = main_window_name
        self.control_window_name = control_window_name
        self.show_overlay = True
        self.tk_root = None
        self.tk_closed = False
        self.vars = {}
        self.screen_width = 1280
        self.screen_height = 720
        self.main_frame = None
        
    def create_windows(self):
        """Cria e configura a janela OpenCV e a interface Tkinter."""
        self.tk_root = tk.Tk()
        self.tk_root.title(self.control_window_name)
        self.tk_root.protocol("WM_DELETE_WINDOW", self.on_tk_close)
        
        # Pega a resolução da tela
        self.screen_width = self.tk_root.winfo_screenwidth()
        self.screen_height = self.tk_root.winfo_screenheight()
        
        # Posiciona a janela de controle do lado direito
        tk_width = 450
        tk_height = 650
        x_pos = self.screen_width - tk_width - 20
        y_pos = 50
        self.tk_root.geometry(f"{tk_width}x{tk_height}+{x_pos}+{y_pos}")
        
        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        # Ajusta o tamanho da janela do OpenCV proporcionalmente à tela
        cv2.resizeWindow(self.main_window, int(self.screen_width * 0.65), int(self.screen_height * 0.8))

    def on_tk_close(self):
        self.tk_closed = True
        if self.tk_root:
            self.tk_root.destroy()

    def is_open(self):
        """Verifica se ambas as janelas (Tkinter e OpenCV) ainda estão abertas."""
        if self.tk_closed:
            return False
        try:
            if cv2.getWindowProperty(self.main_window, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except cv2.error:
            return False
        return True

    def _create_slider(self, parent, label_text, var_name, from_, to, initial):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        lbl = ttk.Label(frame, text=label_text, width=15)
        lbl.pack(side=tk.LEFT)
        
        val_lbl = ttk.Label(frame, text=str(initial), width=4)
        val_lbl.pack(side=tk.RIGHT)
        
        var = tk.IntVar(value=initial)
        self.vars[var_name] = var
        
        def update_val(val):
            val_lbl.config(text=str(int(float(val))))
            
        scale = ttk.Scale(frame, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL, command=update_val)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def _create_combobox(self, parent, label_text, var_name, values, initial_idx=0):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        lbl = ttk.Label(frame, text=label_text, width=15)
        lbl.pack(side=tk.LEFT)
        
        var = tk.IntVar(value=initial_idx)
        self.vars[var_name] = var
        
        def on_select(event):
            var.set(combo.current())
            
        combo = ttk.Combobox(frame, values=values, state="readonly")
        combo.current(initial_idx)
        combo.bind("<<ComboboxSelected>>", on_select)
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def _create_checkbox(self, parent, label_text, var_name, initial):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        var = tk.IntVar(value=1 if initial else 0)
        self.vars[var_name] = var
        
        chk = ttk.Checkbutton(frame, text=label_text, variable=var)
        chk.pack(side=tk.LEFT)

    def setup_trackbars(self, model, model_idx=0):
        """
        Cria os controles na interface Tkinter específicos para o modelo de segmentação ativo.
        """
        if self.tk_root is None or self.tk_closed:
            return
            
        if self.main_frame is not None:
            self.main_frame.destroy()
            
        self.vars.clear()
        
        self.main_frame = ttk.Frame(self.tk_root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        lbl_title = ttk.Label(self.main_frame, text="Painel de Controle", font=("Helvetica", 14, "bold"))
        lbl_title.pack(pady=(10, 0))
        
        lbl_sub = ttk.Label(self.main_frame, text="Ajuste os valores para sintonizar a contagem")
        lbl_sub.pack(pady=(0, 10))
        
        models_list = ["Watershed", "Erosão", "Dilatação"]
        self._create_combobox(self.main_frame, "Modelo", "Modelo", models_list, model_idx)
        
        max_mode = len(model.display_modes) - 1
        display_modes_list = [f"Modo {i} - {model.display_modes[i]}" for i in range(max_mode + 1)]
        self._create_combobox(self.main_frame, "Exibir", "Exibir", display_modes_list, max_mode)
        
        ttk.Separator(self.main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=10)
        
        from models.watershed_model import WatershedModel
        from models.erosion_model import ErosionModel
        from models.dilation_model import DilationModel
        
        if isinstance(model, WatershedModel):
            self._create_combobox(self.main_frame, "Mascara", "Mascara", ["0", "1", "2"], model.params['mask_source'])
            self._create_combobox(self.main_frame, "Filtro", "Filtro", ["0", "1", "2"], model.params['filter_type'])
            
            initial_blur = (model.params['blur_size'] - 1) // 2
            self._create_slider(self.main_frame, "Blur", "Blur", 0, 15, initial_blur)
            
            self._create_checkbox(self.main_frame, "Otsu", "Otsu", model.params['use_otsu'])
            self._create_slider(self.main_frame, "Limiar", "Limiar", 0, 255, model.params['manual_thresh'])
            
            initial_fill = model.params['fill_holes_area'] // 50
            self._create_slider(self.main_frame, "Buracos", "Buracos", 0, 100, initial_fill)
            
            self._create_slider(self.main_frame, "Erosao Sz", "Erosao Sz", 0, 15, model.params['erosion_size'])
            self._create_slider(self.main_frame, "Erosao It", "Erosao It", 0, 10, model.params['erosion_iters'])
            self._create_slider(self.main_frame, "Dilat Sz", "Dilat Sz", 0, 15, model.params['dilation_size'])
            self._create_slider(self.main_frame, "Dilat It", "Dilat It", 0, 10, model.params['dilation_iters'])
            self._create_slider(self.main_frame, "Dist Min", "Dist Min", 0, 100, model.params['min_distance'])
            
        elif isinstance(model, ErosionModel):
            self._create_checkbox(self.main_frame, "Adaptativo", "Adapt.", model.params['use_adaptive'])
            
            initial_block = (model.params['block_size'] - 3) // 2
            self._create_slider(self.main_frame, "Bloco", "Bloco", 0, 49, initial_block)
            
            self._create_slider(self.main_frame, "Const. C", "Const. C", 0, 100, model.params['c_val'])
            
            self._create_checkbox(self.main_frame, "Otsu", "Otsu", model.params['use_otsu'])
            self._create_slider(self.main_frame, "Limiar", "Limiar", 0, 255, model.params['manual_thresh'])
            
            self._create_slider(self.main_frame, "Erosao Sz", "Erosao Sz", 0, 15, model.params['kernel_size'])
            self._create_slider(self.main_frame, "Erosao It", "Erosao It", 0, 10, model.params['erosion_iters'])
            
        elif isinstance(model, DilationModel):
            self._create_checkbox(self.main_frame, "Otsu", "Otsu", model.params['use_otsu'])
            self._create_slider(self.main_frame, "Limiar", "Limiar", 0, 255, model.params['manual_thresh'])
            
            self._create_slider(self.main_frame, "Dilat Sz", "Dilat Sz", 0, 15, model.params['kernel_size'])
            self._create_slider(self.main_frame, "Dilat It", "Dilat It", 0, 10, model.params['dilation_iters'])
            
        self.tk_root.update()

    def update_params_from_trackbars(self, model):
        """
        Lê os controles Tkinter e atualiza os parâmetros do modelo ativo.
        """
        if self.tk_closed:
            return 0, 0
            
        # Atualiza eventos do Tkinter
        self.tk_root.update()
        
        selected_model_idx = self.vars["Modelo"].get() if "Modelo" in self.vars else 0
        display_mode = self.vars["Exibir"].get() if "Exibir" in self.vars else 0
        
        from models.watershed_model import WatershedModel
        from models.erosion_model import ErosionModel
        from models.dilation_model import DilationModel
        
        if isinstance(model, WatershedModel):
            if "Mascara" in self.vars: model.set_param('mask_source', self.vars["Mascara"].get())
            if "Filtro" in self.vars: model.set_param('filter_type', self.vars["Filtro"].get())
            if "Blur" in self.vars: model.set_param('blur_size', (self.vars["Blur"].get() * 2) + 1)
            if "Otsu" in self.vars: model.set_param('use_otsu', self.vars["Otsu"].get() == 1)
            if "Limiar" in self.vars: model.set_param('manual_thresh', self.vars["Limiar"].get())
            if "Buracos" in self.vars: model.set_param('fill_holes_area', self.vars["Buracos"].get() * 50)
            if "Erosao Sz" in self.vars: model.set_param('erosion_size', self.vars["Erosao Sz"].get())
            if "Erosao It" in self.vars: model.set_param('erosion_iters', max(1, self.vars["Erosao It"].get()))
            if "Dilat Sz" in self.vars: model.set_param('dilation_size', self.vars["Dilat Sz"].get())
            if "Dilat It" in self.vars: model.set_param('dilation_iters', max(1, self.vars["Dilat It"].get()))
            if "Dist Min" in self.vars: model.set_param('min_distance', self.vars["Dist Min"].get())
            
        elif isinstance(model, ErosionModel):
            if "Adapt." in self.vars: model.set_param('use_adaptive', self.vars["Adapt."].get() == 1)
            if "Bloco" in self.vars: model.set_param('block_size', (self.vars["Bloco"].get() * 2) + 3)
            if "Const. C" in self.vars: model.set_param('c_val', self.vars["Const. C"].get())
            if "Otsu" in self.vars: model.set_param('use_otsu', self.vars["Otsu"].get() == 1)
            if "Limiar" in self.vars: model.set_param('manual_thresh', self.vars["Limiar"].get())
            if "Erosao Sz" in self.vars: model.set_param('kernel_size', self.vars["Erosao Sz"].get())
            if "Erosao It" in self.vars: model.set_param('erosion_iters', max(1, self.vars["Erosao It"].get()))
            
        elif isinstance(model, DilationModel):
            if "Otsu" in self.vars: model.set_param('use_otsu', self.vars["Otsu"].get() == 1)
            if "Limiar" in self.vars: model.set_param('manual_thresh', self.vars["Limiar"].get())
            if "Dilat Sz" in self.vars: model.set_param('kernel_size', self.vars["Dilat Sz"].get())
            if "Dilat It" in self.vars: model.set_param('dilation_iters', max(1, self.vars["Dilat It"].get()))
            
        return selected_model_idx, display_mode

    def resize_to_fit(self, image, max_width=None, max_height=None):
        """Redimensiona proporcionalmente a imagem para caber na tela."""
        if max_width is None:
            max_width = int(self.screen_width * 0.70)
        if max_height is None:
            max_height = int(self.screen_height * 0.85)
            
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
        """Fecha todas as janelas gerenciadas pelo OpenCV e Tkinter."""
        self.tk_closed = True
        if self.tk_root:
            self.tk_root.destroy()
        cv2.destroyAllWindows()

