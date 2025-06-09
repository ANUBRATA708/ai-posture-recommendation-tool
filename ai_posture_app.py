import sys
import os

# ✅ Redirect stdout and stderr to null to prevent write errors, but avoid log files
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
import tensorflow as tf

# ✅ Resource path (PyInstaller compatible)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ✅ Load model
try:
    model_path = resource_path("posture_model.h5")
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    messagebox.showerror("Model Load Error", f"Unable to load posture_model.h5\n{e}")
    sys.exit(1)

# ✅ Preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img

# ✅ Predict posture
def predict_posture(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image could not be read.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)[0][0]
        return "Good Posture" if prediction > 0.5 else "Bad Posture", prediction
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Failed to process image:\n{str(e)}")
        raise

# ✅ Recommendation text
def get_recommendation(prediction):
    if prediction <= 0.5:
        return (
            "⚠️ Warning: Bad Posture Detected!\n\n"
            "💡 Recommendations:\n"
            "• Sit upright with your back straight.\n"
            "• Keep your shoulders relaxed.\n"
            "• Feet flat on the floor.\n"
            "• Screen at eye level.\n"
            "• Take breaks every 30 minutes."
        )
    else:
        return "✅ All Right! Your posture looks good."

# ✅ GUI Class
class PostureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧍‍♂️ AI-Assisted Posture Detection")
        self.root.geometry("700x750")
        self.root.configure(bg="#f0f4f8")

        title = tk.Label(root, text="AI Posture Advisor", font=("Helvetica", 26, "bold"),
                         bg="#4e73df", fg="white", pady=10)
        title.pack(fill="x")

        self.img_label = tk.Label(root, bd=2, relief="groove", bg="#f0f4f8")
        self.img_label.pack(pady=20)

        self.result_label = tk.Label(root, text="", font=("Arial", 14),
                                     wraplength=600, bg="#f0f4f8")
        self.result_label.pack(pady=10)

        self.upload_btn = tk.Button(root, text="🖼️ Upload an Image",
                                    command=self.upload_image,
                                    font=("Helvetica", 12, "bold"),
                                    bg="#28a745", fg="white", activebackground="#1e7e34", width=30)
        self.upload_btn.pack(pady=10)

        self.capture_btn = tk.Button(root, text="📸 Capture Real-Time Image",
                                     command=self.capture_image,
                                     font=("Helvetica", 12, "bold"),
                                     bg="#1a73e8", fg="white", activebackground="#1558b0", width=30)
        self.capture_btn.pack(pady=10)

        self.capture_btn.bind("<Enter>", lambda e: self.capture_btn.config(bg="#1558b0"))
        self.capture_btn.bind("<Leave>", lambda e: self.capture_btn.config(bg="#1a73e8"))
        self.upload_btn.bind("<Enter>", lambda e: self.upload_btn.config(bg="#1e7e34"))
        self.upload_btn.bind("<Leave>", lambda e: self.upload_btn.config(bg="#28a745"))

    def display_image(self, img_path):
        img = Image.open(img_path).resize((300, 300))
        img = ImageOps.expand(img, border=8, fill='#4e73df')
        img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=img)
        self.img_label.image = img

    def process_image(self, path):
        try:
            result, prob = predict_posture(path)
            self.display_image(path)
            color = "green" if result == "Good Posture" else "red"
            self.result_label.config(
                text=f"{result} ({prob:.2f})\n\n{get_recommendation(prob)}",
                fg=color
            )
        except Exception:
            pass  # messagebox already shown in predict_posture

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.process_image(file_path)

    def capture_image(self):
        try:
            cap = cv2.VideoCapture(0)
            cv2.namedWindow("Press Space to Capture")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow("Press Space to Capture", frame)
                k = cv2.waitKey(1)
                if k % 256 == 32:
                    os.makedirs("captured_images", exist_ok=True)
                    img_name = "captured_images/realtime_test.jpg"
                    cv2.imwrite(img_name, frame)
                    break
            cap.release()
            cv2.destroyAllWindows()
            self.process_image(img_name)
        except Exception as e:
            messagebox.showerror("Camera Error", f"Could not capture image.\n{str(e)}")

# ✅ Launch App
if __name__ == "__main__":
    root = tk.Tk()
    app = PostureApp(root)
    root.mainloop()
