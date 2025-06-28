import os
import sys
import cv2
import numpy as np
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from src.capture import KeypointExtractor


def main(
    max_samples: int = 100,
    threshold: float = 0.2
):
    cap = cv2.VideoCapture(0)
    extractor = KeypointExtractor(mode=True, maxHands=2, detectionCon=0.2, trackCon=0.2)
    samples = []
    print(f"Buscando {max_samples} frames con más de {threshold*100:.0f}% de detección...")

    while len(samples) < max_samples:
        ret, frame = cap.read()
        if not ret:
            break
        kp = extractor.extract(frame)
        ratio = np.count_nonzero(kp) / kp.size
        cv2.putText(frame, f"Ratio: {ratio:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Test Pipeline', frame)
        key = cv2.waitKey(1) & 0xFF
        if ratio >= threshold:
            samples.append(kp)
            print(f"Frame {len(samples)}/{max_samples} guardado (ratio={ratio:.2f})")
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not samples:
        print("No se capturaron frames válidos.")
        return

    arr = np.stack(samples, axis=0)
    out_dir = os.path.join(project_root, 'test_output')
    os.makedirs(out_dir, exist_ok=True)
    path_npy = os.path.join(out_dir, 'samples.npy')
    np.save(path_npy, arr)
    print(f"Guardados {arr.shape[0]} vectores en {path_npy}, shape={arr.shape}")

    nz = int(np.count_nonzero(arr))
    total = int(arr.size)
    global_ratio = nz / total
    summary = {
        'num_samples': arr.shape[0],
        'features_shape': list(arr.shape),
        'global_non_zero': [nz, total],
        'global_non_zero_ratio': round(global_ratio, 4),
        'per_sample_ratio': [round(np.count_nonzero(sample)/sample.size, 4) for sample in samples]
    }
    json_path = os.path.join(out_dir, 'samples.json')
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(summary, jf, indent=2)
    print(f"Guardado resumen en: {json_path}")

if __name__ == '__main__':
    main()
