import tkinter as tk
from tkinter import ttk
from tkinter import Label
from PIL import Image, ImageTk
import cv2
import threading
import time

class UIConfig:
    """Configuration class for UI styles and colors."""
    def __init__(self):
        # Colors 
        self.widget_color = "#003366"  
        self.accent_color = "#d9e3f0"  # statistics section background
        self.button_color = "#003366"  # Consistent with accent color
        self.highlight_color = "#4a90e2"  # For eye-catching highlights

        # Fonts
        self.font = ("Arial", 11)
        self.bold_font = ("Arial", 13, "bold")
        self.title_font = ("Arial", 24, "bold")

class DeceptionDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel AI - Deception Detection")
        self.root.geometry("1400x1000")

        # Load UI configuration
        self.config = UIConfig()

        # Configure the main window
        self.root.configure()

        # Set up custom styles
        self.style = ttk.Style()
        self.style.configure("TFrame", foreground=self.config.widget_color)
        self.style.configure("TButton", foreground=self.config.widget_color)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

        self.captured_images = []

        # Create UI sections
        self.create_title_section()
        self.create_live_video_section()
        self.create_evidence_section()
        self.create_button_and_statistics_section()

        # Start updating the live video feed and statistics bar
        self.start_time = time.time()
        self.show_frame()
        self.update_statistics_bar()

    def create_title_section(self):
        """Creates the title section at the top."""
        title_label = ttk.Label(self.root, padding="0", text="Sentinel AI", font=self.config.title_font, foreground=self.config.widget_color)
        title_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

    def create_live_video_section(self):
        """Creates the live video feed section."""
        live_video_frame = ttk.Frame(self.root, padding="0", style="TFrame")
        live_video_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=(10, 10), sticky="nsew")
        live_video_frame.grid_rowconfigure(0, weight=1)
        live_video_frame.grid_columnconfigure(0, weight=1)

        live_video_label = tk.Label(live_video_frame, text="Live Video Feed", font=self.config.bold_font, fg=self.config.widget_color)
        live_video_label.pack(anchor="nw")

        self.live_video_label = tk.Label(live_video_frame)
        self.live_video_label.pack(fill="both", expand=True) 

    def create_evidence_section(self):
        """Creates the captured deceptive actions section."""
        evidence_frame = ttk.Frame(self.root, padding="0", style="TFrame")
        evidence_frame.grid(row=1, column=2, columnspan=2, padx=10, pady=(10, 10), sticky="nsew")
        evidence_frame.grid_rowconfigure(0, weight=1)
        evidence_frame.grid_columnconfigure(0, weight=1)

        evidence_label = tk.Label(evidence_frame, text="Captured Deceptive Actions", font=self.config.bold_font, fg=self.config.widget_color)
        evidence_label.pack(anchor="nw")

        self.evidence_image = Label(evidence_frame, text="[sample images]", bg=self.config.accent_color)
        self.evidence_image.pack(fill="both", expand=True, pady=10)

    def create_button_and_statistics_section(self):
        """Creates a parent frame for the button and statistics bar sections."""
        button_statistics_frame = ttk.Frame(self.root, padding="0")
        button_statistics_frame.grid(row=1, column=4, columnspan=2, padx=10, pady=(10, 10), sticky="nsew")
        
        # Create buttons
        self.create_button_section(button_statistics_frame)

        # Add spacer frame
        spacer_frame = ttk.Frame(button_statistics_frame, height=250)
        spacer_frame.grid(row=1, column=0)

        # Create statistics section below buttons
        self.create_statistics_section(button_statistics_frame)

        self.create_deception_score_frame(button_statistics_frame)

    def create_button_section(self, parent_frame):
        """Creates the control button section, placed above the statistics bar."""
        button_frame = ttk.Frame(parent_frame, padding="0")
        button_frame.grid(row=0, column=0, padx=10, sticky="nsew")

        start_button = ttk.Button(button_frame, text="Start Analysis", command=self.start_analysis, style="TButton")
        start_button.grid(row=0, column=0, padx=20, pady=(10, 5))

        save_button = ttk.Button(button_frame, text="Prediction History", command=self.view_history, style="TButton")
        save_button.grid(row=1, column=0, padx=20, pady=(10, 10))

    def create_statistics_section(self, parent_frame):
        """Creates the statistics bar section."""
        statistics_frame = ttk.Frame(parent_frame, padding="0", style="TFrame", relief="sunken", borderwidth=1)
        statistics_frame.grid(row=4, column=0, padx=10, pady=(10, 10), sticky="nsew")
        statistics_frame.grid_rowconfigure(0, weight=1)
        statistics_frame.grid_columnconfigure(0, weight=1)

        statistics_label = tk.Label(statistics_frame, text="System Statistics", font=self.config.bold_font, fg=self.config.widget_color)
        statistics_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="nw")

        # Create statistics variables
        self.create_statistics_variables(statistics_frame)

    def create_statistics_variables(self, statistics_frame):
        """Creates the statistics variables and labels."""
        self.processing_time_var = tk.StringVar(value="0.00s")
        self.distance_var = tk.StringVar(value="N/A")
        self.pupil_var = tk.StringVar(value="Not Detected")
        self.face_var = tk.StringVar(value="Not Detected")
        self.body_var = tk.StringVar(value="Not Detected")
        self.effective_detection_var = tk.StringVar(value="N/A")
        self.deceptive_actions_var = tk.StringVar(value="0")

        labels = [
            ("Processing Time", self.processing_time_var),
            ("Distance", self.distance_var),
            ("Pupils", self.pupil_var),
            ("Face", self.face_var),
            ("Body", self.body_var),
            ("Effective Detection", self.effective_detection_var),
            ("Deceptive Actions", self.deceptive_actions_var),
        ]

        for i, (text, var) in enumerate(labels):
            label = tk.Label(statistics_frame, text=f"{text}: ", font=self.config.font, fg=self.config.widget_color)
            label.grid(row=i + 1, column=0, sticky=tk.W, padx=10, pady=5)
            value_label = tk.Label(statistics_frame, textvariable=var, font=self.config.font, fg=self.config.highlight_color)
            value_label.grid(row=i + 1, column=1, sticky=tk.E, padx=5, pady=5)

    def create_deception_score_frame(self, parent_frame):
        """Creates the separate frame for Deception Confidence Score."""
        score_frame = tk.Frame(parent_frame, bg=self.config.accent_color, relief="raised", bd=1)
        score_frame.grid(row=5, column=0, padx=10, pady=(10, 10), sticky="nsew")
        score_frame.grid_rowconfigure(0, weight=1)
        score_frame.grid_columnconfigure(0, weight=1)

        score_label = tk.Label(score_frame, text="Deception Confidence Score", font=self.config.bold_font, bg=self.config.accent_color, fg=self.config.widget_color)
        score_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="nw")

        # Create deception confidence score variable
        self.deception_score_var = tk.StringVar(value="0%")
        deception_score_value = tk.Label(score_frame, textvariable=self.deception_score_var, font=self.config.font, bg=self.config.accent_color, fg=self.config.highlight_color)
        deception_score_value.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

    def start_analysis(self):
        print("Starting Analysis...")  # Replace with actual analysis logic
        self.capture_mock_images()

    def capture_mock_images(self):
        """Simulate capturing deceptive images and update the evidence section."""
        # List of mock image paths (replace with actual paths to your images)
        mock_images = [
            "test.png",
            "test2.png",
        ]

        # Simulate capturing a new image and appending it to the list
        for image_path in mock_images:
            self.update_evidence_display(image_path)

    def update_evidence_display(self, image_path):
        """Update the evidence display with a new image."""
        img = Image.open(image_path)
        img = img.resize((100, 100))  # Resize image for display
        img_tk = ImageTk.PhotoImage(img)

        # Create a new label for each captured image
        image_label = Label(self.evidence_image.winfo_toplevel(), image=img_tk, bg=self.config.accent_color)
        image_label.image = img_tk  # Keep a reference to avoid garbage collection
        row = len(self.captured_images)  
        image_label.grid(row=row, column=0, padx=5, pady=(1, 2), sticky="w")

        self.captured_images.append(image_label)

        # Limit the number of displayed images
        if len(self.captured_images) > 10:  
            # Remove the oldest image
            oldest_image_label = self.captured_images.pop(0)  
            oldest_image_label.grid_forget()  

            for index, label in enumerate(self.captured_images):
                label.grid(row=index, column=0, padx=5, pady=(1, 2), sticky="w")  # Update positions

    def view_history(self):
        """Method to view prediction history (dummy function)."""
        print("Viewing prediction history...")  # Replace with actual history viewing logic

    def show_frame(self):
        """Handles live video feed updates."""
        ret, frame = self.cap.read()  # Read a frame from the video capture
        if ret: 
            frame = cv2.resize(frame, (800, 600))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            img = Image.fromarray(frame)  
            img_tk = ImageTk.PhotoImage(image=img)  
            self.live_video_label.imgtk = img_tk  
            self.live_video_label.config(image=img_tk) 
        self.root.after(10, self.show_frame)  # Refresh frame every 10ms

    def update_statistics_bar(self):
        """Updates the statistics bar with AI analysis data."""
        self.processing_time_var.set(f"{time.time() - self.start_time:.2f}s")
        self.distance_var.set("1.5m")  # Placeholder
        self.pupil_var.set("Detected")   # Placeholder
        self.face_var.set("Detected")     # Placeholder
        self.body_var.set("Detected")     # Placeholder
        self.effective_detection_var.set("82%")  # Placeholder
        self.deceptive_actions_var.set("5")      # Placeholder
        self.deception_score_var.set("68.4%")    # Placeholder
        self.root.after(1000, self.update_statistics_bar)  # Update every second

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = DeceptionDetectionUI(root)
    root.mainloop()

    # Release resources
    app.cap.release()
    cv2.destroyAllWindows()