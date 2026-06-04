import os
import cv2
import sys
from dataset_manager import DatasetManager
from gui import ViewerGUI
from models import WatershedModel, ErosionModel, DilationModel

def main():
    dataset_dir = "dataset"
    db = DatasetManager(dataset_dir)
    
    if not db.has_images:
        print(f"[-] Erro: Nenhuma imagem encontrada na pasta '{dataset_dir}'.")
        print("[*] Insira imagens .jpg, .png, etc., nessa pasta para rodar o sistema.")
        sys.exit(1)
        
    models = [
        WatershedModel(),
        ErosionModel(),
        DilationModel()
    ]
    model_names = ["Watershed", "Erosao", "Dilatacao"]
    active_model_idx = 0
    
    gui = ViewerGUI(main_window_name="Contador de Feijoes", control_window_name="Controles de Parametros")
    gui.create_windows()
    
    gui.setup_trackbars(models[active_model_idx], active_model_idx)
    
    print("\n" + "="*70)
    print("      SISTEMA MODULAR DE VISÃO COMPUTACIONAL - CONTAGEM DE FEIJÕES")
    print("="*70)
    print(f"Dataset carregado: {db.total_images} imagens encontradas.")
    print("\nInstruções de Navegação (Mantenha o foco na janela principal):")
    print("  -> D ou Seta Direita / Espaço : Próxima imagem")
    print("  -> A ou Seta Esquerda         : Imagem anterior")
    print("  -> H                          : Alternar o painel inferior de informações")
    print("  -> 1, 2 ou 3                  : Alternar modelos (Watershed, Erosão, Dilatação)")
    print("  -> Q ou ESC                   : Fechar o sistema")
    print("="*70 + "\n")
    
    current_results = None
    last_params_hash = None
    
    while True:
        if last_params_hash is not None:
            if cv2.getWindowProperty(gui.main_window, cv2.WND_PROP_VISIBLE) < 1 or \
               cv2.getWindowProperty(gui.control_window, cv2.WND_PROP_VISIBLE) < 1:
                print("[*] Janela de exibição ou controle fechada pelo usuário. Encerrando o sistema...")
                break

        idx = db.current_index
        filename = db.get_current_filename()
        
        img = db.get_current_image()
        if img is None:
            print(f"[-] Erro ao carregar imagem: {filename}. Pulando...")
            db.next_image()
            continue
            
        selected_model_idx, display_mode = gui.update_params_from_trackbars(models[active_model_idx])
        
        if selected_model_idx != active_model_idx:
            print(f"[*] Modelo alterado via slider para: {model_names[selected_model_idx]}")
            active_model_idx = selected_model_idx
            gui.setup_trackbars(models[active_model_idx], active_model_idx)
            continue
            
        current_params_hash = (idx, active_model_idx, str(models[active_model_idx].params))
        
        if current_params_hash != last_params_hash:
            current_results = models[active_model_idx].process(img)
            last_params_hash = current_params_hash
            
        bean_count = current_results.get('count', 0)
        display_key = f'display_image_{display_mode}'
        
        if display_key in current_results:
            display_img = current_results[display_key]
        else:
            display_img = current_results['original']
            
        if len(display_img.shape) == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            
        display_img_resized, scale = gui.resize_to_fit(display_img, max_width=1280, max_height=720)
        
        gui.show(display_img_resized, f"[{idx+1}/{db.total_images}]")
        
        key_code = cv2.waitKeyEx(30)
        key = key_code & 0xFF
        
        if key == ord('q') or key == ord('Q') or key == 27:
            print("[*] Encerrando o sistema...")
            break
            
        elif key == ord('d') or key == ord('D') or key == 32 or key_code in (2441984, 0x270000, 39):
            db.next_image()
            print(f"[+] Exibindo imagem {db.current_index+1}/{db.total_images}: {db.get_current_filename()}")
            
        elif key == ord('a') or key == ord('A') or key_code in (2424832, 0x250000, 37):
            db.prev_image()
            print(f"[+] Exibindo imagem {db.current_index+1}/{db.total_images}: {db.get_current_filename()}")
            
        elif key == ord('h') or key == ord('H'):
            gui.show_overlay = not gui.show_overlay
            print(f"[*] Menu overlay {'visível' if gui.show_overlay else 'oculto'}.")
            
        elif key == ord('1'):
            if active_model_idx != 0:
                print("[*] Modelo alterado via teclado para: Watershed")
                active_model_idx = 0
                gui.setup_trackbars(models[active_model_idx], active_model_idx)
        elif key == ord('2'):
            if active_model_idx != 1:
                print("[*] Modelo alterado via teclado para: Erosão")
                active_model_idx = 1
                gui.setup_trackbars(models[active_model_idx], active_model_idx)
        elif key == ord('3'):
            if active_model_idx != 2:
                print("[*] Modelo alterado via teclado para: Dilatação")
                active_model_idx = 2
                gui.setup_trackbars(models[active_model_idx], active_model_idx)
            
    gui.close()

if __name__ == "__main__":
    main()