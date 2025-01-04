from TrafficImagePrediction import TrafficImagePrediction as tip
from TrafficVideoPrediction import TrafficVideoPrediction as tvp
import numpy as np
from customtkinter import CTk, CTkFrame, CTkLabel, CTkEntry, CTkButton, CTkComboBox, filedialog, CTkInputDialog
from PIL import Image, ImageTk
import imageio
import threading
import cv2
import collections
import time
from CTkDataVisualizingWidgets import *

MEDIA = "image"
stop_video_flag = False

def body(MEDIA, file_path):
    media = MEDIA

    if media.lower() == "image":
        handle_image(file_path)
    elif media.lower() == "video":
        handle_video(file_path)

def handle_image(image_path):
    model_path = 'traffic_image_classification_model.h5'
    traffic_predictor = tip(model_path)
    result = traffic_predictor.predict_image(image_path)
    bottom_frame.configure(text=f'The image is classified as: {result}')

def handle_video(video_path):
    count = 0
    ncount = 0
    model_path = 'D:\Projects\IBM Internship\IBM Project\.venv/video_classification_model_2.keras'
    traffic_video_predictor = tvp(model_path= model_path)
    results = traffic_video_predictor.predict_video(video_path)
    for result in results:
        if result.lower() == "traffic":
            count += 1
        elif result.lower() == "no traffic":
            ncount += 1

    if count > ncount:
        bottom_frame.configure(text="Video Traffic Pattern Predictions: Traffic")
    elif count < ncount:
        bottom_frame.configure(text="Video Traffic Pattern Predictions: No Traffic")
    else:
        bottom_frame.configure(text="Video Traffic Pattern Predictions: Neutral")

    global graph_button

    graph_button = CTkButton(frame, text='GRAPH', fg_color="#0504AA", hover_color="#2225C1",
                             command=lambda: graphframepop(video_path))
    graph_button.place(x=20, y=screen_height - 150)

def graphframepop(video_path):
    def graphframedestroy(framegraph):
        framegraph.destroy()
        graph_button.destroy()
        frame.pack(fill="both", expand=True)

    frame.pack_forget()
    framegraph = CTkFrame(master=root, bg_color="white", fg_color="white")
    framegraph.pack(fill="both", expand=True)
    CTkButton(framegraph, text='Back', fg_color="#0504AA", hover_color="#2225C1",
              command=lambda: graphframedestroy(framegraph)).place(x=screen_width - 200, y=150)
    threading.Thread(target=graphframe, args=(framegraph, video_path)).start()


import cv2
import numpy as np


def graphframe(window, video_path):
    # Load YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3 (2).cfg")  # Make sure to replace with your YOLO model files
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Set up video capture
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second from video
    frames_per_interval = fps  # Number of frames in each time interval (1 second = fps frames)

    choice = CTkInputDialog(text="1. Per second\n2. Per minute\n3. Per hour\nType in number:",
                            title="Select the time interval for averaging vehicles:")

    # Set the time interval and unit for graph display
    if choice.get_input() == '1':
        time_interval = 1
        time_unit = "sec"
    elif choice.get_input() == '2':
        time_interval = 60
        time_unit = "min"
    elif choice.get_input() == '3':
        time_interval = 3600
        time_unit = "hour"
    else:
        print("Invalid choice. Defaulting to per second.")
        time_interval = 1
        time_unit = "sec"

    vehicle_count_dict = collections.defaultdict(list)  # Store vehicle counts for averaging
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the image for YOLO detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        vehicle_count = 0

        # Iterate through the detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Adjust the confidence threshold as needed
                    # Vehicle class IDs in COCO dataset (range from 2 to 7 for vehicle-related classes)
                    if class_id in [2, 3, 5, 7]:  # 2: car, 3: motorcycle, 5: bus, 7: truck
                        vehicle_count += 1
                        center_x = int(detection[0] * frame.shape[1])
                        center_y = int(detection[1] * frame.shape[0])
                        w = int(detection[2] * frame.shape[1])
                        h = int(detection[3] * frame.shape[0])

                        # Draw rectangle around detected vehicle
                        cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2),
                                      (center_x + w // 2, center_y + h // 2), (0, 0, 255), 2)

        current_time = time.time()
        elapsed_time = current_time - start_time
        interval = int(elapsed_time // time_interval)
        vehicle_count_dict[interval].append(vehicle_count)  # Append vehicle count for the current frame

        # Show the frame with detected vehicles
        cv2.imshow('video', frame)

        if cv2.waitKey(33) == 27:  # Exit if 'ESC' is pressed
            break

    cv2.destroyAllWindows()

    # Calculate average vehicle count per interval
    avg_vehicle_count_dict = {
        f"{interval}": sum(vehicle_counts) / len(vehicle_counts) if vehicle_counts else 0
        for interval, vehicle_counts in sorted(vehicle_count_dict.items())
    }

    # Prepare data for graph
    avg_vehicle_count_list = [avg_count for interval, avg_count in sorted(avg_vehicle_count_dict.items())]

    # Create chart widgets for average vehicle counts
    CTkChart(window, avg_vehicle_count_dict, corner_radius=20, chart_axis_width=3, width=1500, height=300).place(x=10,
                                                                                                                 y=500)
    CTkGraph(window, avg_vehicle_count_list, width=550, height=400, fg_color="#FF7761", graph_color="#FF7761",
             graph_fg_color="#FF5330", title_font_size=30,
             corner_radius=20).place(x=10, y=30)


def combobox_callback(e):
    global MEDIA
    if file_type.get() == "Image":
        MEDIA = "image"
    else:
        MEDIA = "video"

def open_file_dialog():
    if MEDIA == "video":
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    else:
        file_path = filedialog.askopenfilename(filetypes=[("PNG files", ".png"), ("JPEG files", ".jpg;*.jpeg")])

    if file_path:
        file_path_entry.delete(0, "end")
        file_path_entry.insert(0, file_path)
    else:
        file_path_entry.delete(0, "end")
        file_path_entry.configure(placeholder_text="Select the file")

    return file_path

def play_video(video_path, label, root):
    video = imageio.get_reader(video_path)

    def stream():
        try:
            for frame in video.iter_data():
                if stop_video_flag == True:
                    return
                image = Image.fromarray(frame)
                if not (root.winfo_exists() and label.winfo_exists()):
                    break
                image = image.resize((label.winfo_width(), label.winfo_height()), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(image)
                label.configure(image=photo)
                label.image = photo
                root.update()
            if root.winfo_exists() and not stop_video_flag.is_set():
                root.after(0, play_video, video_path, label, root)  # Loop the video
        except Exception as e:
            print(f"Error in stream: {e}")

    threading.Thread(target=stream).start()

def display_image(image_path, label):
    try:
        image = Image.open(image_path)
        if label.winfo_exists():
            image = image.resize((label.winfo_width(), label.winfo_height()), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo)
            label.image = photo
    except Exception as e:
        print(f"Error in display_image: {e}")

def on_submit():
    global stop_video_flag
    stop_video_flag = True  # Stop any ongoing video playback
    file_path = file_path_entry.get()
    bottom_frame.configure(text="Processing...", text_color="black")
    label.configure(image='', text='')  # Clear the previous content
    if MEDIA == "video":
        stop_video_flag = False
        threading.Thread(target=play_video, args=(file_path, label, root)).start()
    else:
        threading.Thread(target=display_image, args=(file_path, label)).start()
    threading.Thread(target=body, args=(MEDIA, file_path)).start()

def set_background_image(root, image_path):
    # Create a frame for the background image
    bg_frame = CTkFrame(master=root, width=root.winfo_screenwidth(), height=root.winfo_screenheight(), fg_color="white")
    bg_frame.place(x=0, y=0, relwidth=1, relheight=1)

    # Load the background image
    background_image = Image.open(image_path)
    background_image = background_image.resize(
        (root.winfo_screenwidth(), root.winfo_screenheight()),
        Image.Resampling.LANCZOS
    )

    # Create a PhotoImage object from the image
    background_photo = ImageTk.PhotoImage(background_image)

    # Create a label with the image
    background_label = CTkLabel(master=bg_frame, image=background_photo, text="")
    background_label.image = background_photo  # Keep a reference to avoid garbage collection
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

if __name__ == "__main__":
    root = CTk()
    root.geometry("800x600")  # Adjust size as needed

    # Set background image
    set_background_image(root, 'C:/Users/ASUS/Downloads/red-traffic-light-flat-illustration_1284-22959.jpg')  # Change this to your image path

    # Your existing code here...
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    root.geometry(f"{screen_width}x{screen_height - 100}+0+0")
    root.title("Traffic Prediction")

    frame = CTkFrame(master=root, bg_color="white", fg_color="white")
    frame.pack(fill="both", expand=True)

    entry_list_frame = CTkFrame(master=frame, width=500, height=700, bg_color="white", fg_color="white")
    entry_list_frame.pack(side="left", padx=100, pady=10)

    entry_label = CTkLabel(master=entry_list_frame, text="Traffic Predictor Entries", font=("Arial", 20), text_color="black")
    entry_label.place(x=1, y=10)

    file_type = CTkComboBox(master=entry_list_frame, values=["Image", "Video"], button_color="#0504AA", text_color="black",
                            fg_color="white", bg_color="black", border_color="#0504AA", dropdown_fg_color="black",
                            dropdown_hover_color="#353839", hover=0, command=combobox_callback)
    file_type.place(x=1, y=50)

    file_path_entry = CTkEntry(master=entry_list_frame, width=410, fg_color="white", border_width=1, text_color="black",
                               placeholder_text="Select the file")
    file_path_entry.place(x=1, y=90)

    file_path_button = CTkButton(master=entry_list_frame, width=20, height=20, text="📂", font=("Arial", 20),
                                 fg_color="#0504AA", hover_color="#2225C1", anchor="center")
    file_path_button.place(x=420, y=90)
    file_path_button.configure(command=open_file_dialog)

    label = CTkLabel(master=frame, width=800, height=600, fg_color="white", bg_color="black", text="")
    label.pack(side="right", padx=100, pady=10)
    label.propagate(False)

    submit_button = CTkButton(master=entry_list_frame, width=20, height=20, text="SUBMIT", font=("Arial", 15),
                              fg_color="#0504AA", corner_radius=50, hover_color="#2225C1", anchor="center")
    submit_button.place(x=160, y=130)
    submit_button.configure(command=on_submit)

    bottom_frame = CTkLabel(master=frame, fg_color="white", bg_color="white", text="", font=("Arial", 20))
    bottom_frame.place(x=100, y=450)

    root.mainloop()
