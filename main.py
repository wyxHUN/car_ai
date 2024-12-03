import tkinter as tk
from tkinter import filedialog
import cv2
import PIL.Image, PIL.ImageTk
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time


class VideoApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg="#1e272e")
        self.window.geometry("1400x900")  # Nagyobb ablak méret

        # Reszponzív elrendezés beállítása
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=2)  # Videó rész szélesebb legyen
        self.window.columnconfigure(1, weight=1)  # Statisztikai rész

        # Bal oldali rész - Videó kijelzés
        self.video_frame = tk.Frame(self.window, bg="#1e272e")
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        self.video_frame.rowconfigure(0, weight=1)
        self.video_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.video_frame, bg="#2f3640", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Jobb oldali rész - Statisztikák és diagram
        self.stats_frame = tk.Frame(self.window, bg="#1e272e")
        self.stats_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.stats_frame.rowconfigure(0, weight=1)
        self.stats_frame.rowconfigure(1, weight=5)
        self.stats_frame.columnconfigure(0, weight=1)

        # Járművek száma címkék
        self.labels_frame = tk.Frame(self.stats_frame, bg="#1e272e")
        self.labels_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        self.label_cars_count = tk.Label(self.labels_frame, text="Cars: 0", bg="#1e272e", fg="white",
                                         font=("Helvetica", 14, "bold"))
        self.label_cars_count.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.label_trucks_count = tk.Label(self.labels_frame, text="Trucks: 0", bg="#1e272e", fg="white",
                                           font=("Helvetica", 14, "bold"))
        self.label_trucks_count.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.label_motorcycles_count = tk.Label(self.labels_frame, text="Motorcycles: 0", bg="#1e272e", fg="white",
                                                font=("Helvetica", 14, "bold"))
        self.label_motorcycles_count.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        # Matplotlib diagram
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_facecolor("#1e272e")
        self.ax.set_title("Number of Vehicles", color="white", fontsize=16, fontweight='bold')
        self.ax.set_xlabel("Vehicle Type", color="white", fontsize=14)
        self.ax.set_ylabel("Count", color="white", fontsize=14)
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.figure.patch.set_facecolor('#1e272e')
        self.bar_chart = FigureCanvasTkAgg(self.figure, self.stats_frame)
        self.bar_chart.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=10)

        # Gombok és vezérlők frame
        self.controls_frame = tk.Frame(self.window, bg="#1e272e")
        self.controls_frame.grid(row=1, column=0, columnspan=2, pady=15, padx=15, sticky="ew")

        # Videó kiválasztása gomb
        self.btn_select_video = tk.Button(self.controls_frame, text="Select Video", width=20, command=self.select_video,
                                          bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
        self.btn_select_video.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Play/Pause gomb
        self.btn_play_pause = tk.Button(self.controls_frame, text="Play", width=10, command=self.play_pause_video,
                                        bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"))
        self.btn_play_pause.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        # Lejátszási sáv (csúszka)
        self.scale = tk.Scale(self.controls_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=800, bg="#1e272e",
                              fg="white", highlightbackground="#1e272e", font=("Helvetica", 10))
        self.scale.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

        # Videó és állapot változók
        self.video_path = None
        self.cap = None
        self.imgtk = None
        self.playing = False
        self.total_frames = 0
        self.current_frame = 0
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.previous_positions = {}  # Járművek előző pozícióinak tárolása
        self.previous_time = None  # Kezdetben None

        self.window.mainloop()

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.playing = False
            self.btn_play_pause.config(text="Play")
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.scale.config(to=self.total_frames)
            self.current_frame = 0
            self.scale.set(0)
            self.previous_time = None  # Reseteljük az időzítőt

    def play_pause_video(self):
        if not self.playing:
            self.playing = True
            self.btn_play_pause.config(text="Pause")
            self.update_frame()
        else:
            self.playing = False
            self.btn_play_pause.config(text="Play")

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened() and self.playing:
            ret, frame = self.cap.read()
            if ret:
                # Csökkentett képkockaszám feldolgozás a teljesítmény növelése érdekében
                if self.current_frame % 2 == 0:
                    self.current_frame += 1
                    frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))

                    # YOLOv5 felismerés
                    results = self.model(frame)
                    detected_objects = results.pandas().xyxy[0]

                    # Járművek számlálása
                    cars_count = 0
                    trucks_count = 0
                    motorcycles_count = 0
                    current_positions = {}

                    for _, row in detected_objects.iterrows():
                        label = row['name']
                        confidence = row['confidence'] * 100
                        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(
                            row['ymax'])

                        if label == 'car':
                            cars_count += 1
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            cv2.putText(frame, f"Car: {confidence:.1f}%", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 0), 2)
                        elif label == 'truck':
                            trucks_count += 1
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                            cv2.putText(frame, f"Truck: {confidence:.1f}%", (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        elif label == 'motorcycle':
                            motorcycles_count += 1
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                            cv2.putText(frame, f"Motorcycle: {confidence:.1f}%", (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    # Címkék frissítése
                    self.label_cars_count.config(text=f"Cars: {cars_count}")
                    self.label_trucks_count.config(text=f"Trucks: {trucks_count}")
                    self.label_motorcycles_count.config(text=f"Motorcycles: {motorcycles_count}")

                    # Diagram frissítése
                    self.ax.clear()
                    self.ax.set_facecolor("#1e272e")
                    self.ax.set_title("Number of Vehicles", color="white", fontsize=16, fontweight='bold')
                    self.ax.set_xlabel("Vehicle Type", color="white", fontsize=14)
                    self.ax.set_ylabel("Count", color="white", fontsize=14)
                    self.ax.tick_params(axis='x', colors='white')
                    self.ax.tick_params(axis='y', colors='white')
                    vehicle_types = ["Car", "Truck", "Motorcycle"]
                    counts = [cars_count, trucks_count, motorcycles_count]
                    self.ax.bar(vehicle_types, counts, color=['green', 'blue', 'yellow'])
                    self.bar_chart.draw()

                    # OpenCV kép átalakítása Tkinter-hez
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = PIL.Image.fromarray(frame)
                    self.imgtk = PIL.ImageTk.PhotoImage(image=img)
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

                    # Lejátszási sáv frissítése
                    self.scale.set(self.current_frame)

                self.current_frame += 1

            if self.playing:
                self.window.after(10, self.update_frame)
        else:
            if self.cap is not None and not self.cap.isOpened():
                self.cap.release()


# Tkinter ablak létrehozása és alkalmazás futtatása
root = tk.Tk()
app = VideoApp(root, "Video Processing System")
