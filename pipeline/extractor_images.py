#!/usr/bin/env python3
"""
extractor_images.py

Script para extraer features de landmarks de manos desde imágenes estáticas (.jpg/.png).
Diseñado específicamente para procesar datasets como ASL que contienen imágenes individuales
en lugar de secuencias de video.
"""
import os
import sys
import cv2
import numpy as np
import json
from src.capture import KeypointExtractor


def extract_from_images(images_dir: str, 
                       output_dir: str,
                       label: str = None,
                       min_detection_ratio: float = 0.6):
    """
    Extrae features de landmarks de manos desde un directorio de imágenes.
    
    Args:
        images_dir: Directorio que contiene las imágenes (.jpg/.png)
        output_dir: Directorio donde guardar los .npy resultantes
        label: Etiqueta para las imágenes (si None, usa nombre del directorio)
        min_detection_ratio: Ratio mínimo de landmarks detectados para considerar válida la imagen
    
    Returns:
        Número de imágenes procesadas exitosamente
    """
    # Configurar paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    
    # Inicializar extractor
    extractor = KeypointExtractor(mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5)
    hand_dims = 2 * 21 * 4  # 168 features
    
    # Determinar etiqueta
    if label is None:
        label = os.path.basename(images_dir).upper()
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar archivos de imagen
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"⚠️  No se encontraron imágenes en {images_dir}")
        return 0
    
    print(f"Procesando {len(image_files)} imágenes de etiqueta '{label}'...")
    
    valid_features = []
    processed_count = 0
    
    for img_name in sorted(image_files):
        img_path = os.path.join(images_dir, img_name)
        
        # Cargar imagen
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ❌ No se pudo cargar: {img_name}")
            continue
        
        # Extraer keypoints
        try:
            kp = extractor.extract(img)
            kp_hands = kp[:hand_dims]
            
            # Verificar calidad de detección
            ratio = np.count_nonzero(kp_hands) / kp_hands.size
            
            if ratio >= min_detection_ratio:
                valid_features.append(kp_hands)
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"  ✅ Procesadas {processed_count} imágenes...")
            else:
                print(f"  ⚠️  Baja detección ({ratio:.2f}): {img_name}")
                
        except Exception as e:
            print(f"  ❌ Error procesando {img_name}: {e}")
            continue
    
    # Guardar features si hay datos válidos
    if valid_features:
        arr = np.stack(valid_features, axis=0)  # (N_imgs, 168)
        
        # Generar nombre de archivo
        timestamp = __import__('time').strftime('%Y%m%d_%H%M%S')
        npy_name = f"{label}_{timestamp}.npy"
        npy_path = os.path.join(output_dir, npy_name)
        
        np.save(npy_path, arr)
        print(f"✅ Guardado: {npy_path} con {arr.shape[0]} muestras")
        
        # Generar resumen
        nz = int(np.count_nonzero(arr))
        total = int(arr.size)
        global_ratio = nz / total
        
        summary = {
            'nombre_archivo': npy_name,
            'num_muestras': arr.shape[0],
            'forma_caracteristicas': list(arr.shape),
            'no_ceros_global': [nz, total],
            'ratio_no_ceros_global': round(global_ratio, 4) * 100,
            'ratio_no_ceros_por_muestra': [
                round(np.count_nonzero(sample)/sample.size, 4) * 100 
                for sample in valid_features
            ]
        }
        
        json_name = f"{label}_{timestamp}_summary.json"
        json_path = os.path.join(output_dir, json_name)
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(summary, jf, indent=2)
        print(f"✅ Resumen guardado: {json_path}")
        
        return processed_count
    else:
        print(f"❌ No se generaron features válidas para {label}")
        return 0


def process_asl_dataset(asl_root: str, output_base: str):
    """
    Procesa un dataset completo de ASL con estructura:
    ASL_Dataset/Train/A/, ASL_Dataset/Train/B/, etc.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(project_root, 'data', 'features_asl')
    os.makedirs(output_dir, exist_ok=True)
    
    train_dir = os.path.join(asl_root, 'Train')
    if not os.path.isdir(train_dir):
        print(f"❌ No se encontró directorio Train en {asl_root}")
        return
    
    # Listar subdirectorios (letras)
    letters = [d for d in os.listdir(train_dir) 
               if os.path.isdir(os.path.join(train_dir, d))]
    
    labels_list = []
    total_processed = 0
    
    print(f"Encontradas {len(letters)} categorías: {sorted(letters)}")
    
    for letter in sorted(letters):
        print(f"\n=== Procesando categoría: {letter} ===")
        
        letter_input_dir = os.path.join(train_dir, letter)
        letter_output_dir = os.path.join(output_dir, letter.lower())
        
        # Extraer features de esta categoría
        count = extract_from_images(
            images_dir=letter_input_dir,
            output_dir=letter_output_dir,
            label=letter.lower(),
            min_detection_ratio=0.3
        )
        
        if count > 0:
            # Buscar archivos .npy generados para agregar a labels.json
            for fname in os.listdir(letter_output_dir):
                if fname.endswith('.npy'):
                    rel_path = os.path.relpath(
                        os.path.join(letter_output_dir, fname), 
                        project_root
                    )
                    labels_list.append({
                        'filename': rel_path,
                        'label': letter.lower()
                    })
        
        total_processed += count
    
    # Guardar labels.json consolidado
    labels_path = os.path.join(output_dir, 'labels.json')
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(labels_list, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Procesamiento completado!")
    print(f"Total de imágenes procesadas: {total_processed}")
    print(f"Labels guardados en: {labels_path}")
    print(f"Archivos .npy generados: {len(labels_list)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extrae features de landmarks desde imágenes estáticas"
    )
    parser.add_argument("input", help="Directorio de imágenes o dataset ASL")
    parser.add_argument("-o", "--output", help="Directorio de salida para .npy")
    parser.add_argument("-l", "--label", help="Etiqueta para las imágenes")
    parser.add_argument("--asl-dataset", action="store_true",
                        help="Procesar como dataset ASL completo")
    parser.add_argument("--min-ratio", type=float, default=0.3,
                        help="Ratio mínimo de detección (default: 0.3)")
    
    args = parser.parse_args()
    
    if args.asl_dataset:
        # Procesar dataset completo
        process_asl_dataset(args.input, args.output)
    else:
        # Procesar directorio individual
        if not args.output:
            print("❌ Se requiere --output para procesamiento individual")
            return
        
        count = extract_from_images(
            images_dir=args.input,
            output_dir=args.output,
            label=args.label,
            min_detection_ratio=args.min_ratio
        )
        
        if count > 0:
            print(f"✅ Procesamiento completado: {count} imágenes")
        else:
            print("❌ No se procesaron imágenes exitosamente")


if __name__ == '__main__':
    main()
