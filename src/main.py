import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))
import cv2
from src.capture import KeypointExtractor
from src.recorder import ClipRecorder
from training.model import GestureClassifier

hand_dims = 2 * 21 * 4

def main():
    project_root = os.path.abspath(os.path.join(__file__, '..', '..'))
    clips_dir = os.path.join(project_root, 'data', 'clips')
    model_path = os.path.join(project_root, 'training', 'model.joblib')
    
    cap = cv2.VideoCapture(0)
    extractor = KeypointExtractor(maxHands=2)
    recorder = ClipRecorder(output_dir=clips_dir, max_frames=80)
    classifier = GestureClassifier(model_path=model_path)
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
                feat = kp[:hand_dims].reshape(1, -1)
                letter = classifier.predict(feat)[0]
            except Exception as e:
                print(f"Error en predicción (solo manos): {e}")
                letter = '?'

            cv2.putText(frame, str(letter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
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
