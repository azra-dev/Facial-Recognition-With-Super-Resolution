from flet import *
import flet as ft
import cv2
import numpy as np
import base64
import time
import os
import csv
from datetime import datetime

from facenet_prototype import Facenet
from sr_prototype import SR
from picamera2 import Picamera2

FN = Facenet()
GFP = SR()

picam2 = Picamera2()
picam2.preview_configuration.main.size = (400, 300)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

deb_capt = False

def main(page:Page):
    page.window_height = 420
    page.window_width = 540
    page.update()

    # Current Frame -------------------
    captured_frame = ft.Image(
        src="placeholder.jpg",
        width=360,
        height=240,
        fit=ft.ImageFit.CONTAIN
    )

    # Iterating Capture Frames --------
    def capture_frame():
        clear_button.disabled = True
        recognize_button.disabled = False
        update_database_button.disabled = False
        page.update()
        try:
            while True:
                frame=picam2.capture_array()
                if frame is not None:
                    _, buffer = cv2.imencode('.png', frame)
                    png_as_text = base64.b64encode(buffer).decode('utf-8') 
                    captured_frame.src_base64 = png_as_text
                    captured_frame.update()
                    if deb_capt:
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        _, buffer = cv2.imencode('.png', gray_frame)
                        png_as_text = base64.b64encode(buffer).decode('utf-8') 
                        captured_frame.src_base64 = png_as_text
                        captured_frame.update()
                        break
        except Exception as e:
            print(f"Error: {e}")
    
    # Capture Frames
    def trigger_capture(e):
        picam2.stop()
        picam2.preview_configuration.main.size = (4056, 3040)
        picam2.configure("preview")
        picam2.start()
        frame=picam2.capture_array()
        _, buffer = cv2.imencode('.png', frame)
        png_as_text = base64.b64encode(buffer).decode('utf-8') 
        src_base64_img = png_as_text

        picam2.stop()
        picam2.preview_configuration.main.size = (400, 300)
        picam2.configure("preview")
        picam2.start()

        if (src_base64_img is not None):
            if src_base64_img.startswith('data:image'):
                src_base64_img = src_base64_img.split(',')[1]
            
            img_data = base64.b64decode(src_base64_img)
            np_img = np.frombuffer(img_data, dtype=np.uint8)
            conv_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if (database_mode.value):
                capture_path = os.path.join('database', f"{sample_name.value}.png")
                cv2.imwrite(capture_path, conv_img)
            else:
                capture_path = os.path.join('captures', 'standard', f"{sample_name.value}_{variant_id.value}.png")
                cv2.imwrite(capture_path, conv_img)
            

    # Recognize Face
    def trigger_recognize(e):
        global deb_capt
        src_base64_img = captured_frame.src_base64
        if (src_base64_img is not None):
            deb_capt = True
            recognize_button.disabled = True
            update_database_button.disabled = True
            clear_button.disabled = True
            page.update()

            if src_base64_img.startswith('data:image'):
                src_base64_img = src_base64_img.split(',')[1]
            
            img_data = base64.b64decode(src_base64_img)
            np_img = np.frombuffer(img_data, dtype=np.uint8)
            conv_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            cv2.imwrite('set_input/capture.png', conv_img)

            FN.run_recognition()
            show_result()

    def show_result():
        if os.path.exists("set_output/capture_recognition.png"):
            captured_frame.src = "set_output/capture_recognition.png"
            captured_frame.src_base64 = None

        recognize_button.disabled = True
        update_database_button.disabled = True
        clear_button.disabled = False
        captured_frame.update()
        page.update()

    def modify_database(e):
        recognize_button.disabled = True
        update_database_button.disabled = True
        page.update()
        os.remove("known_embeddings.pkl")
        os.remove("known_labels.pkl")
        FN.process_database()
        print("finish updating database")
        recognize_button.disabled = False
        update_database_button.disabled = False
        page.update()

    def clear(e):
        global deb_capt
        captured_frame.src = "placeholder.jpg"
        captured_frame.src_base64 = None
        clear_button.disabled = True
        page.update()
        deb_capt = False
        capture_frame()

    # COMPONENTS
    recognize_button = ft.ElevatedButton("Recognize Faces", on_click=trigger_recognize, disabled=True)
    enhanced_button = ft.ElevatedButton("Enhanced Faces", on_click=trigger_enhance, disabled=False)
    update_database_button = ft.ElevatedButton("Update Database", on_click=modify_database, disabled=True)
    database_mode = ft.Checkbox(value=False, width=20, height=20)

    capture_button = ft.ElevatedButton("Capture Faces", on_click=trigger_capture, disabled=False)
    clear_button = ft.ElevatedButton("Clear", on_click=clear, disabled=True)

    sample_name = ft.TextField(hint_text="Name of Participant", width=200, height=30, text_size=8, text_align=True)
    variant_id = ft.TextField(hint_text="Variant ID", width=100, height=30, text_size=8, dense=True)

    # RACKS
    function_rack = ft.Row([recognize_button, update_database_button, database_mode], alignment=ft.MainAxisAlignment.CENTER)   
    control_rack = ft.Row([capture_button], alignment=ft.MainAxisAlignment.END)
    field_rack = ft.Row([sample_name, variant_id], alignment=ft.MainAxisAlignment.CENTER)

    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.add(captured_frame, function_rack, field_rack, control_rack)
    capture_frame()

ft.app(target=main)