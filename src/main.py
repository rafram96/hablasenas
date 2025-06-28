import cv2
from capture import KeypointExtractor
from src.recorder import ClipRecorder
from model import GestureClassifier


def main():
    cap = cv2.VideoCapture(0)
    extractor = KeypointExtractor(maxHands=2)
    recorder = ClipRecorder(output_dir="../data/clips", max_frames=80)
    classifier = GestureClassifier(model_path="model.joblib")
    try:
        classifier.load()
    except Exception:
        print("No se encontró un modelo entrenado. Ejecuta el entrenamiento primero.")
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
                letter = classifier.predict(kp)
            except Exception:
                letter = '?'
            cv2.putText(frame, str(letter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 3)
        elif mode == 'record':
            recorder.add_frame(frame)
            cv2.putText(frame,
                        f"Grabando {len(recorder.frames)}/{recorder.max_frames}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        cv2.imshow('LSP Translator', frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
