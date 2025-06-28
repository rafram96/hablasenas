import cv2
import numpy as np
from capture import KeypointExtractor

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
    import argparse
    parser = argparse.ArgumentParser(description='Procesa un clip para extraer keypoints.')
    parser.add_argument('clip', help='Ruta al archivo .avi de entrada')
    parser.add_argument('out_npy', help='Ruta de salida .npy para features')
    parser.add_argument('--annotated', help='Ruta de vídeo anotado opcional', default=None)
    args = parser.parse_args()
    process_clip(args.clip, args.out_npy, args.annotated)