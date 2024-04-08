import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")

        self.video_frame = tk.Label(root)
        self.video_frame.pack()

        self.button_load = tk.Button(root, text="Load Video", command=self.load_video)
        self.button_load.pack()

        self.button_play = tk.Button(root, text="Play", command=self.play_video)
        self.button_play.pack()

        self.button_stop = tk.Button(root, text="Stop", command=self.stop_video)
        self.button_stop.pack()

        self.video_path = None
        self.cap = None

    def load_video(self):
        # Modifică metoda load_video pentru a selecta un fișier video aleatoriu
        video_files = [f for f in os.listdir("tiktok") if f.endswith((".mp4", ".avi"))]
        if video_files:
            self.video_path = os.path.join("tiktok", random.choice(video_files))
            self.cap = cv2.VideoCapture(self.video_path)

    def play_video(self):
        if self.cap is not None:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)

                self.video_frame.config(image=frame)
                self.video_frame.image = frame

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

    def stop_video(self):
        if self.cap is not None:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayerApp(root)
    root.mainloop()
