import cv2
import mediapipe as mp

#  Preparación del modelo
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Cargar el video
cap = cv2.VideoCapture("video/testing.mp4")

# Configurar la salida del video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Cuantificar cantidad de movimientos por area
# left_hand(lh) right_hand (rh) both_hands(bh) hands_down(hd)
lh, rh, bh, hd = 0, 0, 0, 0

# Procesar los fotogramas del video
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # Leer un fotograma del video
        ret, frame = cap.read()

        if not ret:
            break

        # Convertir la imagen a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar las poses de la persona en la imagen
        results = pose.process(image)

        # Obtener las coordenadas de los puntos de referencia de las poses
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            landmarks = None

        # Determinar si uno o ambos brazos están levantados
        if landmarks is not None:
            left_arm_up = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y < landmarks[
                mp_pose.PoseLandmark.LEFT_ELBOW].y > landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_arm_up = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y < landmarks[
                mp_pose.PoseLandmark.RIGHT_ELBOW].y > landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y

            if left_arm_up and not right_arm_up:
                lh += 1
                label = f"Brazo izquierdo levantado {lh}"
            elif not left_arm_up and right_arm_up:
                rh += 1
                label = f"Brazo derecho levantado {rh}"
            elif left_arm_up and right_arm_up:
                bh += 1
                label = f"Ambos brazos levantados {bh}"
            else:
                hd += 1
                label = f"Ambos brazos abajo {hd}"

            # Dibujar las poses en la imagen
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Mostrar la etiqueta correspondiente en la imagen
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 250), 2)

        # Escribir el fotograma procesado en el archivo de salida
        out.write(frame)

        # Mostrar la imagen procesada en tiempo real
        cv2.imshow("Mediapipe Body Detection", frame)

        # Salir del loop si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar los recursos
cap.release()
out.release()
cv2.destroyAllWindows()

# Area de Analisis de datos (Pintar Valores)
print("--------TOTAL DE MOVIMIENTOS CUANTIFICADOS-----------------")
print(f"-------Brazo Izquierdo: {lh} Total de veces Levantado-----")
print(f"-------Brazo Derecho:   {rh} Total de veces Levantado-----")
print(f"-------Ambos Brazos:    {bh} Total de veces Levantados----")
print(f"-------Brazos Abajos:   {hd} Total de veces abajo---------")
