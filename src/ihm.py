import customtkinter as ctk
import serial
import threading
import time

# Configuration de l'apparence
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MNISTMonitor(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MNIST - ESP32 AI Monitor")
        self.geometry("700x480")
        
        # --- CONFIGURATION SÉRIE ---
        # /!\ Change "COM3" par ton port réel (ex: "COM5" ou "/dev/ttyUSB0")
        self.serial_port = "COM9" 
        self.baudrate = 115200
        self.ser = None
        
        # Structure de la fenêtre
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- ZONE GAUCHE : MATRICE 28x28 ---
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.canvas_size = 320
        self.canvas = ctk.CTkCanvas(
            self.left_frame, 
            width=self.canvas_size, 
            height=self.canvas_size, 
            bg="white", 
            highlightthickness=0
        )
        self.canvas.pack(expand=True, padx=10, pady=10)
        
        self.label_matrix = ctk.CTkLabel(self.left_frame, text="Vue Prétraitée (Input IA)", font=("Roboto", 14, "italic"))
        self.label_matrix.pack(pady=5)

        # --- ZONE DROITE : RÉSULTATS ---
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.label_title = ctk.CTkLabel(self.right_frame, text="Résultat CNN", font=("Roboto", 24, "bold"))
        self.label_title.pack(pady=(20, 10))

        # Affichage du chiffre prédit
        self.res_var = ctk.StringVar(value="?")
        self.label_res = ctk.CTkLabel(
            self.right_frame, 
            textvariable=self.res_var, 
            font=("Roboto", 120, "bold"), 
            text_color="#3a7ebf"
        )
        self.label_res.pack()

        # Barre de progression / Confiance
        self.prob_var = ctk.StringVar(value="Confiance : --- %")
        self.label_prob = ctk.CTkLabel(self.right_frame, textvariable=self.prob_var, font=("Roboto", 20))
        self.label_prob.pack(pady=5)

        # Temps d'inférence
        self.time_var = ctk.StringVar(value="Inférence : --- ms")
        self.label_time = ctk.CTkLabel(self.right_frame, textvariable=self.time_var, font=("Roboto", 16, "italic"), text_color="gray")
        self.label_time.pack(pady=10)

        # Status Bar
        self.status_var = ctk.StringVar(value="Déconnexion")
        self.status_label = ctk.CTkLabel(self, textvariable=self.status_var, font=("Roboto", 12))
        self.status_label.grid(row=1, column=0, columnspan=2, pady=5)

        # Lancement de la lecture série
        self.start_serial()

    def start_serial(self):
        """Initialise la connexion série dans un thread séparé."""
        def connect():
            try:
                self.ser = serial.Serial(self.serial_port, self.baudrate, timeout=0.1)
                self.status_var.set(f"Connecté sur {self.serial_port}")
                self.read_loop()
            except Exception as e:
                self.status_var.set(f"Erreur : {str(e)}")
                time.sleep(2)
                self.start_serial() # Re-tentative

        threading.Thread(target=connect, daemon=True).start()

    def draw_matrix(self, data):
        """Dessine la matrice 28x28 sur le canvas."""
        self.canvas.delete("all")
        pixel_size = self.canvas_size / 28
        
        for i, val in enumerate(data):
            try:
                v = float(val)
                # Inversion : 0.0 (Arduino) -> 255 (Blanc), 1.0 (Arduino) -> 0 (Noir)
                color_int = int(255 - (v * 255))
                # Format Hexadécimal pour le canvas
                hex_color = f'#{color_int:02x}{color_int:02x}{color_int:02x}'
                
                y = i // 28
                x = i % 28
                
                self.canvas.create_rectangle(
                    x * pixel_size, y * pixel_size, 
                    (x + 1) * pixel_size, (y + 1) * pixel_size, 
                    fill=hex_color, outline=""
                )
            except:
                continue

    def read_loop(self):
        """Boucle de lecture des messages arrivant de l'ESP32."""
        while True:
            if self.ser and self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                    
                    # 1. Réception de la matrice
                    if line.startswith("MAT:"):
                        raw_values = line.replace("MAT:", "").split(",")
                        if len(raw_values) == 784:
                            self.after(0, self.draw_matrix, raw_values)
                    
                    # 2. Réception des résultats (Chiffre, Probabilité, Temps)
                    if line.startswith("RES:"):
                        parts = line.replace("RES:", "").split(",")
                        if len(parts) == 3:
                            val_pred, val_prob, val_time = parts
                            self.after(0, self.res_var.set, val_pred)
                            self.after(0, self.prob_var.set, f"Confiance : {val_prob}%")
                            self.after(0, self.time_var.set, f"Inférence : {val_time} ms")
                            
                except Exception as e:
                    print(f"Erreur de lecture : {e}")

if __name__ == "__main__":
    app = MNISTMonitor()
    app.mainloop()