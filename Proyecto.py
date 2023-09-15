import cv2
import face_recognition
import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Crear una carpeta para almacenar las fotos
output_folder = "fotos_personas"
os.makedirs(output_folder, exist_ok=True)

# Inicializar la cámara
video_capture = cv2.VideoCapture(0)

# Inicializar el diccionario de personas
personas = {}

# Configurar el tamaño de la imagen capturada
video_capture.set(3, 320)  # Ancho
video_capture.set(4, 240)  # Alto

# Contador de fotogramas para limitar la frecuencia de reconocimiento
frame_count = 0

def guardar_foto():
    ret, frame = video_capture.read()
    
    # Encontrar todas las caras en el fotograma
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if face_encodings:
        name = entry_name.get()
        
        if name:
            # Construir el nombre de archivo de la foto
            photo_counter = len([f for f in os.listdir(output_folder) if f.endswith('.jpg')])
            photo_name = f"{name}_foto_{photo_counter + 1}.jpg"
            photo_path = os.path.join(output_folder, photo_name)
            
            # Guardar la foto en la carpeta
            cv2.imwrite(photo_path, frame)
            
            # Guardar las características faciales en el diccionario
            personas[name] = face_encodings[0]
            
            messagebox.showinfo("Éxito", f"Foto de {name} guardada como {photo_name}")
        else:
            messagebox.showerror("Error", "Debes ingresar un nombre antes de guardar la foto.")
    else:
        messagebox.showerror("Error", "No se detectó una cara en la imagen.")

def detectar_caras():
    global frame_count
    ret, frame = video_capture.read()
    
    # Incrementar el contador de fotogramas
    frame_count += 1
    
    # Detectar y reconocer caras cada tercer fotograma
    if frame_count % 3 == 0:
        # Redimensionar la imagen para el reconocimiento (puedes ajustar el tamaño)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Encontrar todas las caras en la imagen redimensionada
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        # Recorrer todas las caras encontradas
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Convertir las características faciales de las personas conocidas en una lista
            known_face_encodings = list(personas.values())
            
            # Comparar esta cara con las caras conocidas
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Desconocido"

            if True in matches:
                index = matches.index(True)
                name = list(personas.keys())[index]

            # Escalar las coordenadas de la cara para mostrarlas en la imagen original
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Dibujar un cuadro alrededor de la cara y mostrar el nombre
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Mostrar el fotograma resultante
    cv2.imshow('Reconocimiento Facial', frame)

# Crear la ventana de la interfaz
root = tk.Tk()
root.title("Reconocimiento Facial")

# Centrar la ventana en la pantalla
window_width = 480
window_height = 280
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Captura de imagen
frame_capture = ttk.LabelFrame(root, text="Captura de Imagen")
frame_capture.grid(column=0, row=0, padx=10, pady=10, sticky="w")
label_name = ttk.Label(frame_capture, text="Nombre:")
label_name.grid(column=0, row=0, padx=10, pady=10, sticky="w")
entry_name = ttk.Entry(frame_capture)
entry_name.grid(column=1, row=0, padx=10, pady=10, sticky="w")
button_capture = ttk.Button(frame_capture, text="Capturar", command=guardar_foto)
button_capture.grid(column=0, row=1, columnspan=2, padx=10, pady=10, sticky="w")

# Detección de rostros
frame_detection = ttk.LabelFrame(root, text="Detección de Rostros")
frame_detection.grid(column=0, row=1, padx=10, pady=10, sticky="w")
button_detect = ttk.Button(frame_detection, text="Detectar Rostros", command=detectar_caras)
button_detect.grid(column=0, row=0, padx=10, pady=10, sticky="w")

# Iniciar el bucle principal de la interfaz
root.mainloop()

# Liberar la cámara y cerrar la ventana cuando finalices
video_capture.release()
cv2.destroyAllWindows()
