import base64
import io
import json
import mimetypes  # Add this import
import os
import subprocess
import time
import uuid
import warnings
from pathlib import Path
from typing import Callable, Tuple

import customtkinter as ctk
import cv2
import requests
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from PIL import Image, ImageOps
from cv2_enumerate_cameras import enumerate_cameras  # Add this import
from dotenv import load_dotenv
from openai import OpenAI
from stability_sdk import client

import modules.globals
import modules.metadata
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    add_blank_map,
    has_valid_map,
    simplify_maps,
)
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    is_image,
    is_video,
    resolve_relative_path,
    has_image_extension,
)

url = "https://open-api-fractal.unisat.io/v2/inscribe/order/create"

ROOT = None
POPUP = None
POPUP_LIVE = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 850

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 960
PREVIEW_DEFAULT_HEIGHT = 540

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = (740,)
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = (890,)
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_MAX_HEIGHT = 100
MAPPER_PREVIEW_MAX_WIDTH = 100

DEFAULT_BUTTON_WIDTH = 200
DEFAULT_BUTTON_HEIGHT = 40

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None
popup_status_label = None
popup_status_label_live = None
source_label_dict = {}
source_label_dict_live = {}
target_label_dict_live = {}

img_ft, vid_ft = modules.globals.file_types

load_dotenv()

data = json.load(open("nft_response.json"))


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def save_switch_states():
    switch_states = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box
    }
    with open("switch_states.json", "w") as f:
        json.dump(switch_states, f)


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            switch_states = json.load(f)
        modules.globals.keep_fps = switch_states.get("keep_fps", True)
        modules.globals.keep_audio = switch_states.get("keep_audio", True)
        modules.globals.keep_frames = switch_states.get("keep_frames", False)
        modules.globals.many_faces = switch_states.get("many_faces", False)
        modules.globals.map_faces = switch_states.get("map_faces", False)
        modules.globals.color_correction = switch_states.get("color_correction", False)
        modules.globals.nsfw_filter = switch_states.get("nsfw_filter", False)
        modules.globals.live_mirror = switch_states.get("live_mirror", False)
        modules.globals.live_resizable = switch_states.get("live_resizable", False)
        modules.globals.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.mouth_mask = switch_states.get("mouth_mask", False)
        modules.globals.show_mouth_mask_box = switch_states.get("show_mouth_mask_box", False)
    except FileNotFoundError:
        # If the file doesn't exist, use default values
        pass


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label, show_fps_switch

    load_switch_states()

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(
        f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
    )
    root.configure()
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    # Create the main form
    create_image_generation_form(root)

    source_label = ctk.CTkLabel(root, text=None)
    # source_label.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)

    target_label = ctk.CTkLabel(root, text=None)
    # target_label.place(relx=0.6, rely=0.1, relwidth=0.3, relheight=0.25)

    select_face_button = ctk.CTkButton(
        root, text="Select a face", cursor="hand2", command=lambda: select_source_path()
    )
    # select_face_button.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

    swap_faces_button = ctk.CTkButton(
        root, text="↔", cursor="hand2", command=lambda: swap_faces_paths()
    )
    # swap_faces_button.place(relx=0.45, rely=0.4, relwidth=0.1, relheight=0.1)

    select_target_button = ctk.CTkButton(
        root,
        text="Select a target",
        cursor="hand2",
        command=lambda: select_target_path(),
    )
    # select_target_button.place(relx=0.6, rely=0.4, relwidth=0.3, relheight=0.1)

    keep_fps_value = ctk.BooleanVar(value=modules.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(
        root,
        text="Keep fps",
        variable=keep_fps_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_fps", keep_fps_value.get()),
            save_switch_states(),
        ),
    )
    # keep_fps_checkbox.place(relx=0.1, rely=0.6)

    keep_frames_value = ctk.BooleanVar(value=modules.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(
        root,
        text="Keep frames",
        variable=keep_frames_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_frames", keep_frames_value.get()),
            save_switch_states(),
        ),
    )
    # keep_frames_switch.place(relx=0.1, rely=0.65)

    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui["face_enhancer"])
    enhancer_switch = ctk.CTkSwitch(
        root,
        text="Face Enhancer",
        variable=enhancer_value,
        cursor="hand2",
        command=lambda: (
            update_tumbler("face_enhancer", enhancer_value.get()),
            save_switch_states(),
        ),
    )
    # enhancer_switch.place(relx=0.1, rely=0.7)

    keep_audio_value = ctk.BooleanVar(value=modules.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(
        root,
        text="Keep audio",
        variable=keep_audio_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_audio", keep_audio_value.get()),
            save_switch_states(),
        ),
    )
    # keep_audio_switch.place(relx=0.6, rely=0.6)

    many_faces_value = ctk.BooleanVar(value=modules.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(
        root,
        text="Many faces",
        variable=many_faces_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "many_faces", many_faces_value.get()),
            save_switch_states(),
        ),
    )
    # many_faces_switch.place(relx=0.6, rely=0.65)

    color_correction_value = ctk.BooleanVar(value=modules.globals.color_correction)
    color_correction_switch = ctk.CTkSwitch(
        root,
        text="Fix Blueish Cam",
        variable=color_correction_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "color_correction", color_correction_value.get()),
            save_switch_states(),
        ),
    )
    # color_correction_switch.place(relx=0.6, rely=0.7)
    map_faces = ctk.BooleanVar(value=modules.globals.map_faces)
    map_faces_switch = ctk.CTkSwitch(
        root,
        text="Map faces",
        variable=map_faces,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "map_faces", map_faces.get()),
            save_switch_states(),
        ),
    )
    # map_faces_switch.place(relx=0.1, rely=0.75)

    show_fps_value = ctk.BooleanVar(value=modules.globals.show_fps)
    show_fps_switch = ctk.CTkSwitch(
        root,
        text="Show FPS",
        variable=show_fps_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "show_fps", show_fps_value.get()),
            save_switch_states(),
        ),
    )
    # show_fps_switch.place(relx=0.6, rely=0.75)

    mouth_mask_var = ctk.BooleanVar(value=modules.globals.mouth_mask)
    mouth_mask_switch = ctk.CTkSwitch(
        root,
        text="Mouth Mask",
        variable=mouth_mask_var,
        cursor="hand2",
        command=lambda: setattr(modules.globals, "mouth_mask", mouth_mask_var.get()),
    )
    # mouth_mask_switch.place(relx=0.1, rely=0.55)

    show_mouth_mask_box_var = ctk.BooleanVar(value=modules.globals.show_mouth_mask_box)
    show_mouth_mask_box_switch = ctk.CTkSwitch(
        root,
        text="Show Mouth Mask Box",
        variable=show_mouth_mask_box_var,
        cursor="hand2",
        command=lambda: setattr(
            modules.globals, "show_mouth_mask_box", show_mouth_mask_box_var.get()
        ),
    )
    # show_mouth_mask_box_switch.place(relx=0.6, rely=0.55)

    start_button = ctk.CTkButton(
        root, text="Start", cursor="hand2", command=lambda: analyze_target(start, root)
    )
    # start_button.place(relx=0.15, rely=0.80, relwidth=0.2, relheight=0.05)

    stop_button = ctk.CTkButton(
        root, text="Destroy", cursor="hand2", command=lambda: destroy()
    )
    # stop_button.place(relx=0.4, rely=0.80, relwidth=0.2, relheight=0.05)

    preview_button = ctk.CTkButton(
        root, text="Preview", cursor="hand2", command=lambda: toggle_preview()
    )
    # preview_button.place(relx=0.65, rely=0.80, relwidth=0.2, relheight=0.05)

    # --- Camera Selection ---
    camera_label = ctk.CTkLabel(root, text="Select Camera:")
    # camera_label.place(relx=0.1, rely=0.86, relwidth=0.2, relheight=0.05)

    # available_cameras = get_available_cameras()
    # Convert camera indices to strings for CTkOptionMenu
    # available_camera_indices, available_camera_strings = available_cameras
    # camera_variable = ctk.StringVar(
    #     value=(
    #         available_camera_strings[0]
    #         if available_camera_strings
    #         else "No cameras found"
    #     )
    # )
    # camera_optionmenu = ctk.CTkOptionMenu(
    #     root, variable=camera_variable, values=available_camera_strings
    # )
    # camera_optionmenu.place(relx=0.35, rely=0.86, relwidth=0.25, relheight=0.05)

    # live_button = ctk.CTkButton(
    #     root,
    #     text="Live",
    #     cursor="hand2",
    #     command=lambda: webcam_preview(
    #         root,
    #         available_camera_indices[
    #             available_camera_strings.index(camera_variable.get())
    #         ],
    #     ),
    # )
    # live_button.place(relx=0.65, rely=0.86, relwidth=0.2, relheight=0.05)
    # --- End Camera Selection ---

    status_label = ctk.CTkLabel(root, text=None, justify="center")
    # status_label.place(relx=0.1, rely=0.9, relwidth=0.8)

    donate_label = ctk.CTkLabel(
        root, text="Deep Live Cam", justify="center", cursor="hand2"
    )
    # donate_label.place(relx=0.1, rely=0.95, relwidth=0.8)

    return root


def analyze_target(start: Callable[[], None], root: ctk.CTk):
    if POPUP != None and POPUP.winfo_exists():
        update_status("Please complete pop-up or close it.")
        return

    if modules.globals.map_faces:
        modules.globals.souce_target_map = []

        if is_image(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_image()
        elif is_video(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_video()

        if len(modules.globals.souce_target_map) > 0:
            create_source_target_popup(start, root, modules.globals.souce_target_map)
        else:
            update_status("No faces found in target")
    else:
        select_output_path(start)


def create_source_target_popup(
        start: Callable[[], None], root: ctk.CTk, map: list
) -> None:
    global POPUP, popup_status_label

    POPUP = ctk.CTkToplevel(root)
    POPUP.title("Source x Target Mapper")
    POPUP.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}")
    POPUP.focus()

    def on_submit_click(start):
        if has_valid_map():
            POPUP.destroy()
            select_output_path(start)
        else:
            update_pop_status("Atleast 1 source with target is required!")

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP, width=POPUP_SCROLL_WIDTH, height=POPUP_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    def on_button_click(map, button_num):
        map = update_popup_source(scrollable_frame, map, button_num)

    for item in map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text="Select source image",
            command=lambda id=id: on_button_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=0, padx=50, pady=10)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        image = Image.fromarray(cv2.cvtColor(item["target"]["cv2"], cv2.COLOR_BGR2RGB))
        image = image.resize(
            (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        tk_image = ctk.CTkImage(image, size=image.size)

        target_image = ctk.CTkLabel(
            scrollable_frame,
            text=f"T-{id}",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        target_image.grid(row=id, column=3, padx=10, pady=10)
        target_image.configure(image=tk_image)

    popup_status_label = ctk.CTkLabel(POPUP, text=None, justify="center")
    popup_status_label.grid(row=1, column=0, pady=15)

    close_button = ctk.CTkButton(
        POPUP, text="Submit", command=lambda: on_submit_click(start)
    )
    close_button.grid(row=2, column=0, pady=10)


def update_popup_source(
        scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global source_label_dict

    source_path = ctk.filedialog.askopenfilename(
        title="select an source image",
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in map[button_num]:
        map[button_num].pop("source")
        source_label_dict[button_num].destroy()
        del source_label_dict[button_num]

    if source_path == "":
        return map
    else:
        cv2_img = cv2.imread(source_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["source"] = {
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                cv2.cvtColor(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict[button_num] = source_image
        else:
            update_pop_status("Face could not be detected in last upload!")
        return map


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title("Preview")
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value)
    )

    return preview


def update_status(text: str) -> None:
    try:
        status_label.configure(text=text)
        ROOT.update()
    except:
        pass


def update_pop_status(text: str) -> None:
    popup_status_label.configure(text=text)


def update_pop_live_status(text: str) -> None:
    popup_status_label_live.configure(text=text)


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, update the frame processors
    if PREVIEW.state() == "normal":
        global frame_processors
        frame_processors = get_frame_processors_modules(
            modules.globals.frame_processors
        )


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(
        title="select an source image",
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )
    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)


def swap_faces_paths() -> None:
    global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET

    source_path = modules.globals.source_path
    target_path = modules.globals.target_path

    if not is_image(source_path) or not is_image(target_path):
        return

    modules.globals.source_path = target_path
    modules.globals.target_path = source_path

    RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
    RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)

    PREVIEW.withdraw()

    source_image = render_image_preview(modules.globals.source_path, (200, 200))
    source_label.configure(image=source_image)

    target_image = render_image_preview(modules.globals.target_path, (200, 200))
    target_label.configure(image=target_image)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(
        title="select an target image or video",
        initialdir=RECENT_DIRECTORY_TARGET,
        filetypes=[img_ft, vid_ft],
    )
    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        image = render_image_preview(modules.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        modules.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title="save image output file",
            filetypes=[img_ft],
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title="save video output file",
            filetypes=[vid_ft],
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        start()


def check_and_ignore_nsfw(target, destroy: Callable = None) -> bool:
    """Check if the target is NSFW.
    TODO: Consider to make blur the target.
    """
    from numpy import ndarray
    from modules.predicter import predict_image, predict_video, predict_frame

    if type(target) is str:  # image/video file path
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif type(target) is ndarray:  # frame object
        check_nsfw = predict_frame
    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(
                to_quit=False
            )  # Do not need to destroy the window frame if the target is NSFW
        update_status("Processing ignored!")
        return True
    else:
        return False


def fit_image_to_size(image, width: int, height: int):
    if width is None and height is None:
        return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    if width > height:
        ratio_h = height / h
    else:
        ratio_w = width / w
    ratio = max(ratio_w, ratio_h)
    new_size = (int(ratio * w), int(ratio * h))
    return cv2.resize(image, dsize=new_size)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(
        video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == "normal":
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        for frame_processor in get_frame_processors_modules(
                modules.globals.frame_processors
        ):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(modules.globals.source_path)), temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        update_status("Processing succeed!")
        PREVIEW.deiconify()


def webcam_preview(root: ctk.CTk, camera_index: int):
    if not modules.globals.map_faces:
        if modules.globals.source_path is None:
            # No image selected
            return
        create_webcam_preview(camera_index)
    else:
        modules.globals.souce_target_map = []
        create_source_target_popup_for_webcam(
            root, modules.globals.souce_target_map, camera_index
        )


def get_available_cameras():
    """Returns a list of available camera names and indices."""
    camera_indices = []
    camera_names = []

    for camera in enumerate_cameras():
        cap = cv2.VideoCapture(camera.index)
        if cap.isOpened():
            camera_indices.append(camera.index)
            camera_names.append(camera.name)
            cap.release()
    return (camera_indices, camera_names)


def create_webcam_preview(camera_index: int):
    global preview_label, PREVIEW

    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_DEFAULT_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_DEFAULT_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, 60)

    preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)

    PREVIEW.deiconify()

    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)

    source_image = None
    prev_time = time.time()
    fps_update_interval = 0.5  # Update FPS every 0.5 seconds
    frame_count = 0
    fps = 0

    while camera:
        ret, frame = camera.read()
        if not ret:
            break

        temp_frame = frame.copy()

        if modules.globals.live_mirror:
            temp_frame = cv2.flip(temp_frame, 1)

        if modules.globals.live_resizable:
            temp_frame = fit_image_to_size(
                temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
            )

        if not modules.globals.map_faces:
            if source_image is None and modules.globals.source_path:
                source_image = get_one_face(cv2.imread(modules.globals.source_path))

            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui["face_enhancer"]:
                        temp_frame = frame_processor.process_frame(None, temp_frame)
                else:
                    temp_frame = frame_processor.process_frame(source_image, temp_frame)
        else:
            modules.globals.target_path = None

            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui["face_enhancer"]:
                        temp_frame = frame_processor.process_frame_v2(temp_frame)
                else:
                    temp_frame = frame_processor.process_frame_v2(temp_frame)

        # Calculate and display FPS
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= fps_update_interval:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time

        if modules.globals.show_fps:
            cv2.putText(
                temp_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageOps.contain(
            image, (temp_frame.shape[1], temp_frame.shape[0]), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        ROOT.update()

        if PREVIEW.state() == "withdrawn":
            break

    camera.release()
    PREVIEW.withdraw()


def create_source_target_popup_for_webcam(
        root: ctk.CTk, map: list, camera_index: int
) -> None:
    global POPUP_LIVE, popup_status_label_live

    POPUP_LIVE = ctk.CTkToplevel(root)
    POPUP_LIVE.title("Source x Target Mapper")
    POPUP_LIVE.geometry(f"{POPUP_LIVE_WIDTH}x{POPUP_LIVE_HEIGHT}")
    POPUP_LIVE.focus()

    def on_submit_click():
        if has_valid_map():
            POPUP_LIVE.destroy()
            simplify_maps()
            create_webcam_preview(camera_index)
        else:
            update_pop_live_status("At least 1 source with target is required!")

    def on_add_click():
        add_blank_map()
        refresh_data(map)
        update_pop_live_status("Please provide mapping!")

    popup_status_label_live = ctk.CTkLabel(POPUP_LIVE, text=None, justify="center")
    popup_status_label_live.grid(row=1, column=0, pady=15)

    add_button = ctk.CTkButton(POPUP_LIVE, text="Add", command=lambda: on_add_click())
    add_button.place(relx=0.2, rely=0.92, relwidth=0.2, relheight=0.05)

    close_button = ctk.CTkButton(
        POPUP_LIVE, text="Submit", command=lambda: on_submit_click()
    )
    close_button.place(relx=0.6, rely=0.92, relwidth=0.2, relheight=0.05)


def refresh_data(map: list):
    global POPUP_LIVE

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP_LIVE, width=POPUP_LIVE_SCROLL_WIDTH, height=POPUP_LIVE_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    def on_sbutton_click(map, button_num):
        map = update_webcam_source(scrollable_frame, map, button_num)

    def on_tbutton_click(map, button_num):
        map = update_webcam_target(scrollable_frame, map, button_num)

    for item in map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text="Select source image",
            command=lambda id=id: on_sbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=0, padx=30, pady=10)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        button = ctk.CTkButton(
            scrollable_frame,
            text="Select target image",
            command=lambda id=id: on_tbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=3, padx=20, pady=10)

        if "source" in item:
            image = Image.fromarray(
                cv2.cvtColor(item["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=id, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)

        if "target" in item:
            image = Image.fromarray(
                cv2.cvtColor(item["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=id, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)


def update_webcam_source(
        scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global source_label_dict_live

    source_path = ctk.filedialog.askopenfilename(
        title="select an source image",
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in map[button_num]:
        map[button_num].pop("source")
        source_label_dict_live[button_num].destroy()
        del source_label_dict_live[button_num]

    if source_path == "":
        return map
    else:
        cv2_img = cv2.imread(source_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["source"] = {
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                cv2.cvtColor(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict_live[button_num] = source_image
        else:
            update_pop_live_status("Face could not be detected in last upload!")
        return map


def update_webcam_target(
        scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global target_label_dict_live

    target_path = ctk.filedialog.askopenfilename(
        title="select an target image",
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "target" in map[button_num]:
        map[button_num].pop("target")
        target_label_dict_live[button_num].destroy()
        del target_label_dict_live[button_num]

    if target_path == "":
        return map
    else:
        cv2_img = cv2.imread(target_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["target"] = {
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                cv2.cvtColor(map[button_num]["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=button_num, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)
            target_label_dict_live[button_num] = target_image
        else:
            update_pop_live_status("Face could not be detected in last upload!")
        return map


def create_image_generation_form(root: ctk.CTk):
    """Creates a form to input image generation details."""
    global form_frame, output_frame, num_images_var, custom_instruction_var, image_inputs_var

    # Frame for the form
    form_frame = ctk.CTkFrame(root, width=ROOT_WIDTH, height=200)
    form_frame.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.4)

    # Input: Number of images to generate
    ctk.CTkLabel(form_frame, text="Number of NFTs (Max 10 for now):", anchor="w").grid(row=0, column=0, padx=10, pady=10, sticky="w")
    num_images_var = ctk.IntVar(value=1)
    ctk.CTkEntry(form_frame, textvariable=num_images_var, width=200).grid(row=0, column=1, padx=10, pady=10, sticky="w")

    # Button to dynamically create inputs for image selection
    ctk.CTkButton(form_frame, text="Select Source Image", command=generate_image_input_fields).grid(row=0, column=2, padx=10, pady=10, sticky="w")

    # Input: Custom instruction
    ctk.CTkLabel(form_frame, text="Prompt:", anchor="w").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    custom_instruction_var = ctk.StringVar()
    ctk.CTkEntry(form_frame, textvariable=custom_instruction_var, width=400).grid(row=1, column=1, columnspan=2, padx=10, pady=10, sticky="w")

    # Submit button
    ctk.CTkButton(form_frame, text="Generate NFTs", command=submit_image_generation_form).grid(row=2, column=0, columnspan=3, pady=20, sticky="nsew")

    # Output Frame
    output_frame = ctk.CTkFrame(root, width=ROOT_WIDTH, height=300)
    # output_frame.place(relx=0.05, rely=0.5, relwidth=0.9, relheight=0.4)

    # Add a title for the output frame
    # ctk.CTkLabel(output_frame, text="Generated Images", font=("Arial", 14, "bold"), anchor="w", ).pack(side="top", padx=10, pady=5)


from tkinter import filedialog


def generate_image_input_fields():
    """Dynamically generate input fields for selecting image files."""
    global image_inputs_var, form_frame

    """Open a file dialog to select an image and set its path to the given StringVar."""
    image_inputs_var = []
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        image_var = ctk.StringVar()
        image_var.set(file_path)
        image_inputs_var.append(image_var)

    # # Clear previous inputs
    # for widget in form_frame.grid_slaves():
    #     if int(widget.grid_info()["row"]) > 2:
    #         widget.destroy()
    #
    # num_images = num_images_var.get()
    # if num_images > 10:
    #     num_images = 10  # Enforce maximum limit
    #     num_images_var.set(10)
    #
    # image_inputs_var = []
    # for i in range(num_images):
    #     # Label for image selection
    #     ctk.CTkLabel(form_frame, text=f"Image {i + 1} Path:").grid(
    #         row=3 + i, column=0, padx=10, pady=5
    #     )
    #
    #     # StringVar to hold the selected image path
    #     image_var = ctk.StringVar()
    #     image_inputs_var.append(image_var)
    #
    #     # Entry field to display the selected path
    #     image_entry = ctk.CTkEntry(
    #         form_frame, textvariable=image_var, width=300, state="readonly"
    #     )
    #     image_entry.grid(row=3 + i, column=1, padx=10, pady=5)
    #
    #     # Button to open file dialog for image selection
    #     ctk.CTkButton(
    #         form_frame,
    #         text="Select Image",
    #         command=lambda var=image_var: select_image(var),
    #     ).grid(row=3 + i, column=2, padx=10, pady=5)


def select_image(image_var):
    """Open a file dialog to select an image and set its path to the given StringVar."""
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        image_var.set(file_path)


def submit_image_generation_form():
    """Handle form submission and process image generation."""
    global output_frame

    # Collect inputs
    num_images = num_images_var.get()
    if num_images > 10:
        update_status("Error: Maximum of 10 images can be selected.")
        return

    # Get the selected image paths
    selected_image_paths = [var.get() for var in image_inputs_var if var.get()]
    if len(selected_image_paths) < 1:
        update_status("Error: Please select source image")
        return

    custom_instruction = custom_instruction_var.get()

    print("Generating images... Please wait.")
    print(f"Number of images: {num_images}")
    print(f"Selected image paths: {selected_image_paths}")
    print(f"Custom instruction: {custom_instruction}")

    # Generate images using the script
    generate_images_with_script(num_images, selected_image_paths, custom_instruction)


def create_output_directory():
    """Create a directory in the user's Downloads folder to store output images.

    Returns:
        str: The full path of the created directory.
    """
    # Get the user's Downloads directory
    downloads_dir = str(Path.home() / "Downloads")

    # Create the full path for the output directory
    output_dir = os.path.join(downloads_dir, uuid.uuid4().__str__())

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


# Our Host URL should not be prepended with "https" nor should it have a trailing slash.
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'

# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],  # API Key reference.
    verbose=True,  # Print debug messages.
    engine="stable-diffusion-xl-1024-v1-0",  # Set the engine to use for generation.
    # Check out the following link for a list of available engines: https://platform.stability.ai/docs/features/api-parameters#engine
)

client = OpenAI()


def generate_target_image(output_folder, custom_instruction, source_image, index):
    messages = [
        {
            "role": "system",
            "content": "You are a great artist of the 18th century. Known for the creation of the most beautiful and intricate paintings by imagination of the mind."
        },
        {
            "role": "user",
            "content": "Create a prompt for a LLM model to generate an image of a person in different scenarios and styles like SteamPunk, CyberPunk, Da vinci, Van gogh etc. The Prompt should have detailed description of a human, how they look, what they are wearing, what is the background, etc. Also add instruction to generate clear and complete face of the person. The description for the scenarios should be detailed and should be able to generate a clear image in the mind of the reader."
        },
        {
            "role": "user",
            "content": "IMPORTANT: Generate only the prompt, don't add any kind of header like 'Prompt:' or 'Prompt for Image Generation:'"
        }
    ]

    if len(custom_instruction) > 0:
        messages.append({
            "role": "user",
            "content": "Additional Instructions for Image: " + custom_instruction
        })

    print(f"Generating prompt for target image for source image {source_image}")
    completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages)

    print(f"Generating target image using prompt for source image {source_image}")
    # Set up our initial generation parameters.
    answers = stability_api.generate(
        prompt=completion.choices[0].message.content,
        seed=4253978046,  # If a seed is provided, the resulting generated image will be deterministic.
        # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
        # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
        steps=50,  # Amount of inference steps performed on image generation. Defaults to 30.
        cfg_scale=8.0,  # Influences how strongly your generation is guided to match your prompt.
        # Setting this value higher increases the strength in which it tries to match your prompt.
        # Defaults to 7.0 if not specified.
        width=256,  # Generation width, defaults to 512 if not included.
        height=256,  # Generation height, defaults to 512 if not included.
        samples=1,  # Number of images to generate, defaults to 1 if not included.
        sampler=generation.SAMPLER_K_DPMPP_2M  # Choose which sampler we want to denoise our generation with.
        # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
        # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, save generated images.
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                img.save(os.path.join(output_folder,
                                      f"target_image_{str(artifact.seed)}_{index}.png"))  # Save our generated images with their seed number as the filename.
                return os.path.join(output_folder, f"target_image_{str(artifact.seed)}_{index}.png")

    return None


def generate_images_with_script(num_images, selected_image_paths, custom_instruction):
    """Generate images by running the external script."""
    global output_frame

    # Clear previous outputs
    for widget in output_frame.grid_slaves():
        widget.destroy()

    try:
        # Generate images one by one
        output_images = []
        output_dir = create_output_directory()
        for i in range(num_images):
            source_path = selected_image_paths[0]
            target_path = generate_target_image(output_dir, custom_instruction, source_path, i)
            output_path = os.path.join(output_dir, f"output_image_{i + 1}.png")

            if target_path is None:
                print("Error generating target image, skipping...")
                continue

            # Construct the command
            command = [
                r"venv\Scripts\python",
                "run.py",
                "--source",
                source_path,
                "--target",
                target_path,
                "--output",
                output_path
            ]

            # Run the script
            print(f"Target image generated, now proceeding for face swap between {source_path} and {target_path}")
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise RuntimeError(f"Error running script for Image {i + 1}: {result.stderr.decode('utf-8')}")

            # Image.open(output_path).show()

            print("Creating NFTs")

            # Determine the MIME type based on the file extension
            mime_type, _ = mimetypes.guess_type(output_path)
            if mime_type is None:
                mime_type = "application/octet-stream"  # Fallback MIME type

            # Read and encode the output file to Base64
            try:
                with open(output_path, "rb") as image_file:
                    encoded_bytes = base64.b64encode(image_file.read())
                    encoded_string = encoded_bytes.decode('utf-8')
            except Exception as e:
                print(f"Failed to read or encode {output_path}: {e}")
                continue  # Skip to the next image if encoding fails

            # Create the Data URL
            data_url = f"data:{mime_type};base64,{encoded_string}"

            # Prepare the JSON payload
            payload = json.dumps({
                "receiveAddress": os.environ["NFT_RECEIVE_ADDRESS"],  # Replace with actual address
                "feeRate": 1,
                "outputValue": 546,
                "files": [
                    {
                        "filename": os.path.basename(output_path),  # Dynamically set filename
                        "dataURL": data_url  # Use the generated Data URL
                    }
                ],
                # "devAddress": "string",  # Replace with actual dev address
                # "devFee": 0
            })

            headers = {
                'Content-Type': 'application/json',
                "Authorization": os.environ["NFT_API_KEY"]
            }

            # Send the POST request
            try:
                response = requests.post(url, headers=headers, data=payload)
                response.raise_for_status()  # Raises HTTPError for bad responses
                print(f"NFT created successfully for Image {i + 1}: {response.text}")
                # data.push(json.loads(response.text))
                # json.dump(data, open("data.json", "w"))
            except requests.exceptions.RequestException as e:
                print(f"Failed to create NFT for Image {i + 1}: {e}")

            # new_output_path = os.path.join(output_dir, f"output_image_{i + 1}_enh.png")
            #
            # command = [r"venv\Scripts\python", "run.py", "--source", source_path, "--target", output_path, "--output", new_output_path, '--frame-processor', 'face_enhancer']
            #
            # # Run the script
            # print(f"Face swap between {source_path} and {target_path} completed, now proceeding for face enhancement")
            # result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # if result.returncode != 0:
            #     raise RuntimeError(f"Error running script for Image {i + 1}: {result.stderr.decode('utf-8')}")

            Image.open(output_path).show()
            print(f"Image {i + 1} generated successfully: {output_path}, proceeding to next image...")
            # Save the generated image path
            output_images.append(output_path)

        # Display the generated images
        # for output_image in output_images:
        #     img = Image.open(output_image)
        #     img.show()
    except Exception as e:
        print(f"Error generating images: {str(e)}")
        # ctk.CTkLabel(output_frame, text=f"Error: {str(e)}", text_color="red").grid(row=0, column=0, padx=10, pady=10)
