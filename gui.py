from tkinter import *
from tkinter import font as tkfont
from tkinter import messagebox
import cv2, os
import numpy as np
from PIL import Image,ImageTk
import time
import shutil

path = '.\\myfaces'
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
confidence_limit = 123
recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, confidence_limit)
video_capture = cv2.VideoCapture(0)
listOfProfiles = [[], []]  
indexOfProfile = 0  


def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")


def loadProfiles():
    global indexOfProfile, listOfProfiles
    listOfProfiles = [[], []]  # Reset list để tránh dữ liệu thừa
    if os.path.exists("listOfProfiles.txt"):
        with open("listOfProfiles.txt", "r") as file:
            try:
                indexOfProfile = int(file.readline().strip())
                for _ in range(indexOfProfile):
                    profile_id = file.readline().strip()
                    profile_name = file.readline().strip()
                    if profile_id and profile_name:  # Kiểm tra dữ liệu hợp lệ
                        listOfProfiles[0].append(int(profile_id))
                        listOfProfiles[1].append(profile_name)
            except ValueError:
                print("Error loading profiles: invalid data.")

def saveProfiles():
    global indexOfProfile, listOfProfiles
    with open("listOfProfiles.txt", "w") as file:
        file.write(str(indexOfProfile) + "\n")
        for i in range(indexOfProfile):
            file.write(str(listOfProfiles[0][i]) + "\n" + str(listOfProfiles[1][i]) + "\n")


loadProfiles()
if os.path.exists("trainer.yml"):
    recognizer.read("trainer.yml")  

def get_images(path):
    images = []
    labels = []

    for profile_name in os.listdir(path):
        profile_path = os.path.join(path, profile_name)
        if os.path.isdir(profile_path):
            for image_file in os.listdir(profile_path):
                image_path = os.path.join(profile_path, image_file)

                gray = Image.open(image_path).convert('L')
                image = np.array(gray, 'uint8')

                faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    images.append(image[y: y + h, x: x + w])
                    labels.append(listOfProfiles[1].index(profile_name))

    return images, labels

def show(path):
    global video_capture, faceCascade, recognizer, listOfProfiles
    if not video_capture.isOpened():
        video_capture.open(0)

    while True:
        result, video_frame = video_capture.read()
        if not result:
            print("Camera not working.")
            break

        gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(video_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            try:
                if w > 0 and h > 0:
                    number_predicted, conf = recognizer.predict(gray_frame[y:y + h, x:x + w])

                    if conf <= confidence_limit:
                        profile_name = listOfProfiles[1][number_predicted]
                        print(f"{profile_name} recognized with confidence {conf}")
                        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(video_frame, profile_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        raise ValueError("Unrecognized face")
            except (IndexError, ValueError):
                print("Face not recognized")
                cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Khung màu đỏ
                cv2.putText(video_frame, "Unrecognized", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(video_frame, "Press Q to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Recognizing Face", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def addProfile():
    window.withdraw()  # Ẩn giao diện chính
    profile = Toplevel(window)
    profile.title("Add new profile")
    center_window(profile, 600, 250)  # Đưa giao diện con ra giữa màn hình

    p_frame = Frame(
        profile,
        padx=10,
        pady=10
    )
    p_frame.pack(expand=True)

    p_text = Label(
        p_frame,
        text="Input profile name:"
    )
    p_text.grid(row=1, column=1)

    p_name = Entry(
        p_frame
    )
    p_name.grid(row=2, column=1)

    p_about = Label(
        p_frame,
        text="Input name of new profile, and press continue. Then you'll must show your face for 20 seconds."
    )
    p_about.grid(row=3, column=1)

    p_button = Button(
        p_frame,
        text="Continue",
        command=lambda: saveFaces(p_name.get(), profile)
    )
    p_button.grid(row=5, column=1)

    def on_close():
        window.deiconify()  # Hiện lại giao diện chính
        profile.destroy()

    profile.protocol("WM_DELETE_WINDOW", on_close)

def manageProfiles():
    window.withdraw()  # Ẩn giao diện chính
    manage_window = Toplevel(window)
    manage_window.title("Manage Profiles")
    center_window(manage_window, 400, 400)  # Đưa giao diện con ra giữa màn hình

    profile_frame = Frame(manage_window)
    profile_frame.pack(fill=BOTH, expand=True)

    for i, profile_name in enumerate(listOfProfiles[1]):
        profile_label = Label(profile_frame, text=profile_name)
        profile_label.grid(row=i, column=0, padx=10, pady=5)

        delete_button = Button(
            profile_frame,
            text="Delete",
            command=lambda name=profile_name: deleteProfile(name, manage_window)
        )
        delete_button.grid(row=i, column=1, padx=10, pady=5)

    sync_button = Button(
        profile_frame,
        text="Sync Profiles",
        command=syncProfiles
    )
    sync_button.grid(row=len(listOfProfiles[1]), column=1, padx=10, pady=5)

    def on_close():
        window.deiconify()  # Hiện lại giao diện chính
        manage_window.destroy()

    manage_window.protocol("WM_DELETE_WINDOW", on_close)

def saveFaces(name, window_to_destroy):
    global listOfProfiles, indexOfProfile
    if name in listOfProfiles[1]:
        messagebox.showerror("Error", "Profile name already exists.")
        return

    listOfProfiles[0].append(indexOfProfile)
    listOfProfiles[1].append(name)
    indexOfProfile += 1
    saveProfiles()  # Lưu dữ liệu ngay lập tức

    profile_path = os.path.join(path, name)
    if not os.path.exists(profile_path):
        os.makedirs(profile_path)

    if not video_capture.isOpened():
        video_capture.open(0)

    image_count = 0
    while image_count < 300:  # Chụp đủ 300 ảnh
        result, video_frame = video_capture.read()
        if not result:
            print("Camera error: Unable to capture frame.")
            break

        gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Hiển thị khung xanh xung quanh khuôn mặt
            cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Khung xanh cho khuôn mặt

            # Lưu ảnh khuôn mặt trong thư mục
            filename = os.path.join(profile_path, f"{name}_face_{image_count}.png")
            face_img = gray_frame[y:y + h, x:x + w]  # Chỉ lưu vùng khuôn mặt
            cv2.imwrite(filename, face_img)
            print(f"Saving image {image_count} for {name}")
            image_count += 1

        cv2.putText(video_frame, f"Capturing: {image_count}/300", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Capture Training Data", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Process interrupted by user.")
            break

    video_capture.release()
    cv2.destroyAllWindows()
    saveProfiles()  # Lưu dữ liệu

    if image_count >= 300:
        messagebox.showinfo("Capture Complete", f"Captured {image_count} images for {name}.")
    else:
        messagebox.showwarning("Capture Incomplete", f"Only {image_count} images were captured.")

    # Hiện lại giao diện chính sau khi cửa sổ con đóng
    window.deiconify()
    window_to_destroy.destroy()


def deleteProfile(profile_name, window):
    global listOfProfiles, indexOfProfile
    profile_path = os.path.join(path, profile_name)
    if os.path.isdir(profile_path):
        shutil.rmtree(profile_path)  # Xóa thư mục chứa ảnh khuôn mặt

    if profile_name in listOfProfiles[1]:
        index = listOfProfiles[1].index(profile_name)
        del listOfProfiles[0][index]  # Xóa ID tương ứng
        del listOfProfiles[1][index]  # Xóa tên profile

        # Cập nhật số lượng profiles
        indexOfProfile = len(listOfProfiles[0])

        # Ghi lại dữ liệu vào file
        saveProfiles()

        print(f"Profile '{profile_name}' has been deleted.")
    else:
        print(f"Profile '{profile_name}' not found in the list.")

    window.destroy()
    manageProfiles()



def syncProfiles():
    images, labels = get_images(path)
    if images and labels:
        recognizer.train(images, np.array(labels))
        recognizer.save("trainer.yml")  # Lưu mô hình sau khi huấn luyện
        messagebox.showinfo("Sync Profiles", "Profile data synchronized successfully.")
    else:
        messagebox.showwarning("Sync Profiles", "No new images found for training.")


window = Tk()
window.title("Face Recognizer")
window.resizable(False, False)
center_window(window, 700, 350)
title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
left_frame = Frame(window, padx=20, pady=20)
left_frame.pack(side=LEFT, fill=BOTH, expand=True)
title_label = Label(left_frame, text="Face Recognition System", font=title_font)
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
btn_profile = Button(left_frame, text="Make new profile", command=lambda: addProfile())
btn_profile.grid(row=1, column=0, pady=10, sticky="ew")
btn_camera = Button(left_frame, text="Recognition in real time", command=lambda: show(path))
btn_camera.grid(row=2, column=0, pady=10, sticky="ew")
btn_manage_profiles = Button(left_frame, text="Manage Profiles", command=manageProfiles)
btn_manage_profiles.grid(row=3, column=0, pady=10, sticky="ew")

left_frame.grid_columnconfigure(0, weight=1)

right_frame = Frame(window, padx=20, pady=20)
right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

img = Image.open("homepagepic.png")
img = img.resize((200, 200), Image.LANCZOS)  # Resize ảnh
img_photo = ImageTk.PhotoImage(img)
img_label = Label(right_frame, image=img_photo)
img_label.image = img_photo  # Giữ tham chiếu ảnh để tránh bị xóa
img_label.pack(pady=10)


instructions = Label(
    right_frame, 
    text="Instructions:\n1. Make a new profile.\n2. Use real-time recognition to test.\n3. Manage profiles as needed.",
    justify=LEFT
)
instructions.pack(pady=10)

window.mainloop()
