import cv2
import numpy as np
import os
from src.capture import KeypointExtractor

def process_clip(
    clip_path: str,
    output_features_path: str,
    annotated_video_path: str = None
) -> None:
    """
    Procesa un clip AVI para extraer keypoints de manos y cara.
    - Guarda un array NumPy de forma (num_frames, num_features) en output_features_path.
    - Opcionalmente guarda un vídeo anotado con landmarks en annotated_video_path.
    """
    extractor = KeypointExtractor()
    cap = cv2.VideoCapture(clip_path)
    features = []
    writer = None

    if annotated_video_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Extrae y dibuja landmarks en frame interno
        kp = extractor.extract(frame)
        features.append(kp)
        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()

    arr = np.stack(features, axis=0)
    np.save(output_features_path, arr)
    print(f"Features guardadas en: {output_features_path}")
    if annotated_video_path:
        print(f"Vídeo anotado guardado en: {annotated_video_path}")

if __name__ == '__main__':
    # Carpeta de clips por defecto y salida en 'output' al nivel del proyecto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    clips_dir = os.path.join(project_root, 'data', 'clips')
    out_dir = os.path.join(project_root, 'output')
    annotated_dir = os.path.join(out_dir, 'annotated')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)
    # Procesar todos los clips .avi
    for fname in sorted(os.listdir(clips_dir)):
        if not fname.lower().endswith('.avi'):
            continue
        clip_path = os.path.join(clips_dir, fname)
        stem = os.path.splitext(fname)[0]
        feat_path = os.path.join(out_dir, f"{stem}.npy")
        annot_path = os.path.join(annotated_dir, f"{stem}_annot.avi")
        process_clip(clip_path, feat_path, annot_path)
