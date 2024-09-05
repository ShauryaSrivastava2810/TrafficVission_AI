from customtkinter import CTk, CTkFrame, CTkLabel, CTkEntry, CTkButton, CTkComboBox, filedialog, CTkInputDialog
from PIL import Image, ImageTk
from TrafficImagePrediction import TrafficImagePrediction as tip
from TrafficVideoPrediction import TrafficVideoPrediction as tvp
import imageio
import threading
import cv2
import collections
import time
import requests

MEDIA = "image"
stop_video_flag = False


def __body__(MEDIA, file_path):
    media = MEDIA

    if media.lower() == "image":
        handle_image(file_path)
    elif media.lower() == "video":
        handle_video(file_path)


def handle_image(image_path):
    with open(image_path, 'rb') as img_file:
        response = requests.post("http://localhost:5000/classify", files={'file': img_file},
                                 data={'file_type': 'image'})
    result = response.json().get('result')
    bottom_frame.configure(text=f'The image is classified as: {result}')


def handle_video(video_path):
    with open(video_path, 'rb') as video_file:
        response = requests.post("http://localhost:5000/classify", files={'file': video_file},
                                 data={'file_type': 'video'})
    result = response.json().get('result')
    graph_path = response.json().get('graph_path')
    bottom_frame.configure(text=f'Video Traffic Pattern Predictions: {result}')

    global graph_button

    graph_button = CTkButton(frame, text='GRAPH', fg_color="#0504AA", hover_color="#2225C1",
                             command=lambda: graphframepop(graph_path))
    graph_button.place(x=20, y=screen_height - 150)


def graphframepop(graph_path):
    def graphframedestroy(framegraph):
        framegraph.destroy()
        graph_button.destroy()
        frame.pack(fill="both", expand=True)

    frame.pack_forget()
    framegraph = CTkFrame(master=root, bg_color="black", fg_color="black")
    framegraph.pack(fill="both", expand=True)
    CTkButton(framegraph, text='Back', fg_color="#0504AA", hover_color="#2225C1",
              command=lambda: graphframedestroy(framegraph)).place(x=screen_width - 200, y=150)
    threading.Thread(target=graphframe, args=(framegraph, graph_path)).start()


def graphframe(window, graph_path):
    image = Image.open(graph_path)
    image = image.resize((window.winfo_width(), window.winfo_height()), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    label = CTkLabel(window, image=photo)
    label.image = photo
    label.pack(fill="both", expand=True)


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
        file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")])

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
    bottom_frame.configure(text="Processing...")
    label.configure(image='', text='')  # Clear the previous content
    if MEDIA == "video":
        stop_video_flag = False
        threading.Thread(target=play_video, args=(file_path, label, root)).start()
    else:
        threading.Thread(target=display_image, args=(file_path, label)).start()
    threading.Thread(target=__body__, args=(MEDIA, file_path)).start()


if __name__ == "__main__":
    root = CTk()
    frame = CTkFrame(master=root, bg_color="black", fg_color="black")
    frame.pack(fill="both", expand=True)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    root.geometry(f"{screen_width}x{screen_height - 100}+0+0")
    root.title("Traffic Prediction")

    entry_list_frame = CTkFrame(master=frame, width=500, height=700, bg_color="black", fg_color="black")
    entry_list_frame.pack(side="left", padx=100, pady=10)

    entry_label = CTkLabel(master=entry_list_frame, text="Traffic Predictor Entries", font=("Arial", 20))
    entry_label.place(x=1, y=10)

    file_type = CTkComboBox(master=entry_list_frame, values=["Image", "Video"], button_color="#0504AA",
                            fg_color="black", bg_color="black", border_color="#0504AA", dropdown_fg_color="black",
                            dropdown_hover_color="#353839", hover=0, command=combobox_callback)
    file_type.place(x=1, y=50)

    file_path_entry = CTkEntry(master=entry_list_frame, width=410, fg_color="black", border_width=1,
                               placeholder_text="Select the file")
    file_path_entry.place(x=1, y=90)

    file_path_button = CTkButton(master=entry_list_frame, width=20, height=20, text="📂", font=("Arial", 20),
                                 fg_color="#0504AA", hover_color="#2225C1", anchor="center")
    file_path_button.place(x=420, y=90)
    file_path_button.configure(command=open_file_dialog)

    label = CTkLabel(master=frame, width=800, height=600, fg_color="black", bg_color="black", text="")
    label.pack(side="right", padx=100, pady=10)
    label.propagate(False)

    submit_button = CTkButton(master=entry_list_frame, width=20, height=20, text="SUBMIT", font=("Arial", 15),
                              fg_color="#0504AA", corner_radius=50, hover_color="#2225C1", anchor="center")
    submit_button.place(x=160, y=130)
    submit_button.configure(command=on_submit)

    bottom_frame = CTkLabel(master=frame, fg_color="black", bg_color="black", text="", font=("Arial", 20))
    bottom_frame.place(x=100, y=450)

    root.mainloop()
