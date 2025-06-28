"""
malla_facial.py

Script para mostrar la malla facial completa y resaltar puntos clave en amarillo.
Solo necesita OpenCV y MediaPipe.
Pulsa 'q' para salir.
"""

import cv2
import mediapipe as mp

# Inicialización de MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Índices clave a resaltar
IDXS = [1,4,33,133,61,291]
IMPORTANT_IDXS = [1, 4, 2, 5, 10, 33, 133, 55, 65, 93, 199, 61, 291, 285, 295, 323]

# Especificaciones de dibujo
draw_spec = mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1)
highlight_spec = {'color': (0, 255, 255), 'radius': 3}  # Amarillo

def main():
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convertir a RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Procesar
            results = face_mesh.process(image)
            # Volver a BGR para mostrar
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Dibujar malla facial completa
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=draw_spec,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    # Resaltar puntos clave
                    h, w, _ = image.shape
                    for idx in IMPORTANT_IDXS:
                        lm = face_landmarks.landmark[idx]
                        x_px, y_px = int(lm.x * w), int(lm.y * h)
                        cv2.circle(image, (x_px, y_px), highlight_spec['radius'], highlight_spec['color'], -1)

            cv2.imshow('Face Mesh con Puntos Clave', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
