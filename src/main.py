# Asegurar que el proyecto root esté en sys.path
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

import cv2


from src.capture import KeypointExtractor
from src.recorder import ClipRecorder
from training.model import GestureClassifier


def main():
    # Definir rutas absolutas
    project_root = os.path.abspath(os.path.join(__file__, '..', '..'))
    clips_dir = os.path.join(project_root, 'data', 'clips')
    model_path = os.path.join(project_root, 'training', 'model.joblib')
    
    cap = cv2.VideoCapture(0)
    extractor = KeypointExtractor(maxHands=2)
    recorder = ClipRecorder(output_dir=clips_dir, max_frames=80)
    classifier = GestureClassifier(model_path=model_path)
    # Cargar modelo
    try:
        classifier.load()
    except Exception as e:
        print(f"Error al cargar modelo: {e}\nEjecuta 'python training/train.py' primero.")
        sys.exit(1)
    mode = 'translation'
    print("Presiona 't' para traducir, 'r' para grabar clip, 's' para guardar clip, 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            mode = 'record'
            recorder.frames = []
            print("Modo grabación activado.")
        elif key == ord('t'):
            mode = 'translation'
            print("Modo traducción activado.")
        elif key == ord('s') and mode == 'record':
            recorder.save_clip()
        elif key == ord('q'):
            break

        kp = extractor.extract(frame)
        if mode == 'translation':
            try:
                # Asegurar forma 2D y obtener probabilidades
                # Seleccionar mismas features usadas en entrenamiento (222 dims)
                hand_dims = 2 * 21 * 4  # 168 dims manos
                IMPORTANT_IDXS = [1, 4, 2, 5, 10, 33, 133, 55, 65, 93, 199, 61, 291, 285, 295, 323, 263, 362]
                face_indices = []
                for fi in IMPORTANT_IDXS:
                    base = hand_dims + fi * 3
                    face_indices.extend([base, base + 1, base + 2])
                indices = list(range(hand_dims)) + face_indices
                feat_sel = kp[indices]
                feat = feat_sel.reshape(1, -1)
                
                probas = classifier.predict_proba(feat)[0]
                letter = classifier.predict(feat)[0]
                # Mostrar letra con su confianza máxima en porcentaje
                treshold = 0.8
                confidence = probas.max() * 100
                if confidence < treshold or len(classifier.pipeline.classes_) < 2:
                    display_text = '?'
                else:
                    display_text = f"{letter} ({confidence:.1f}%)"
            except Exception as e:
                print(f"Error en predicción: {e}")
                display_text = '?'
            cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 2)
        elif mode == 'record':
            recorder.add_frame(frame)
            cv2.putText(frame,
                        f"Grabando {len(recorder.frames)}/{recorder.max_frames}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 1)

        cv2.imshow('LSP Translator', frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
