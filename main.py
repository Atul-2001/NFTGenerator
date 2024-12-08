import base64
import io
import json
import mimetypes
import os
import subprocess
import threading
import tkinter as tk
import uuid
import warnings
from pathlib import Path
from tkinter import scrolledtext
from tkinter import ttk, filedialog, messagebox

import requests
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from PIL import Image, ImageTk
from PIL.Image import Resampling
from dotenv import load_dotenv
from openai import OpenAI
from stability_sdk import client

load_dotenv()

data = json.load(open("nft_response.json", "r"))

# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],  # API Key reference.
    verbose=True,  # Print debug messages.
    engine="stable-diffusion-xl-1024-v1-0",  # Set the engine to use for generation.
    # Check out the following link for a list of available engines: https://platform.stability.ai/docs/features/api-parameters#engine
)

openai_client = OpenAI()

url = "https://open-api-fractal.unisat.io/v2/inscribe/order/create"


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


def generate_target_image(output_folder, custom_instruction, source_image, index):
    messages = [
        {
            "role": "system",
            "content": "You are a great artist of the 21st century. Known for the creation of the most beautiful and intricate paintings by imagination of the mind."
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
    completion = openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)

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


class NFTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.selected_image_path = None
        self.nft_responses = []
        self.title("NFT Generator")

        # Set a base dark background
        self.configure(bg="#333333")

        # Configure style for dark theme
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background="#333333", foreground="#FFFFFF")
        style.configure("TFrame", background="#333333")
        style.configure("TButton", background="#555555", foreground="#FFFFFF")
        style.configure("TEntry", fieldbackground="#555555", foreground="#FFFFFF")
        style.configure("TSpinbox", fieldbackground="#555555", foreground="#FFFFFF")
        style.configure("Treeview", background="#555555", foreground="#FFFFFF", fieldbackground="#555555")
        style.map("TButton", background=[("active", "#777777")], foreground=[("active", "#FFFFFF")])

        self.selected_image = None
        self.img_label = None

        # ---- Section 1: Form ----
        form_frame = ttk.Frame(self)
        form_frame.pack(padx=10, pady=10, fill='x')

        # Row 1: No of NFTs + Select Source Image button
        row1_frame = ttk.Frame(form_frame)
        row1_frame.pack(fill='x', pady=5)

        ttk.Label(row1_frame, text="NFT Receive Address:").pack(side='left', padx=(0, 5))
        self.rec_addr_var = tk.StringVar(value=os.environ["NFT_RECEIVE_ADDRESS"])

        # Using a Spinbox for number input
        self.rec_addr_box = ttk.Entry(row1_frame, textvariable=self.rec_addr_var, width=80)
        self.rec_addr_box.pack(side='left', padx=(0, 10))

        ttk.Label(row1_frame, text="No of NFTs(Max 10 for now):").pack(side='left', padx=(0, 5))
        self.nfts_var = tk.StringVar(value="1")

        # Using a Spinbox for number input
        self.nft_spin = ttk.Spinbox(row1_frame, from_=1, to=9999, textvariable=self.nfts_var, width=10)
        self.nft_spin.pack(side='left', padx=(0, 10))

        self.select_image_btn = ttk.Button(row1_frame, text="Select Source Image", command=self.select_image)
        self.select_image_btn.pack(side='left', padx=(0, 10))

        self.generate_btn = ttk.Button(row1_frame, text="Generate NFTs", command=self.generate_nfts)
        self.generate_btn.pack(side='left', padx=(0, 10))

        self.clear_btn = ttk.Button(row1_frame, text="Clear", command=self.clear_fields)
        self.clear_btn.pack(side='left')

        # Row 2: Prompt on left, image on right
        row2_frame = ttk.Frame(form_frame)
        row2_frame.pack(fill='x', pady=5)

        # Image display on the right
        image_frame = ttk.Frame(row2_frame)
        image_frame.pack(side='left', padx=(10, 20))

        self.img_label = ttk.Label(image_frame, text="No image selected", anchor='center', relief='groove')
        self.img_label.config(width=40)  # Removed height
        self.img_label.pack()

        # Prompt area on the left
        prompt_frame = ttk.Frame(row2_frame)
        prompt_frame.pack(side='left', fill='both', expand=True)
        ttk.Label(prompt_frame, text="Prompt:").pack(anchor='w', pady=(0, 5))
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, wrap='word', height=15, bg="#555555", fg="#FFFFFF", insertbackground='white')
        self.prompt_text.pack(fill='both', expand=True)
        # self.prompt_text.insert(tk.END, "Enter your prompt here...")

        # # ---- Section 2: Table ----
        # table_frame = ttk.Frame(self)
        # table_frame.pack(padx=10, pady=10, fill='both', expand=True)
        #
        # columns = ("Image", "PayAddress", "Amount")
        # self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=5)
        #
        # self.tree.heading("#0", text="Image")
        # self.tree.column("#0", width=150)
        #
        # self.tree.heading("PayAddress", text="PayAddress")
        # self.tree.column("PayAddress", width=200)
        #
        # self.tree.heading("Amount", text="Amount")
        # self.tree.column("Amount", width=100)
        #
        # self.tree.pack(fill='both', expand=True)
        #
        # # TODO: Insert initial table data if needed
        # # for example:
        # my_image = Image.open(r"C:\Users\beast\Downloads\bf652f6d-51c1-4b6e-b12c-81fab4000a3b\output_image_1.png")
        # self.tree.insert("", "end", text="", image=my_image, values=("bc1pn2yh86uw8uqde3kn7yjlfnmx97qgzl8vzmjulhxd2820smaat2jswzd662", "134241"))

        # ---- Section 2: Table ----
        table_frame = ttk.Frame(self)
        style.configure("Treeview", rowheight=180)  # Set row height to match your image size
        table_frame.pack(padx=10, pady=10, fill='both', expand=True)

        # Use tree headings to display both tree column (#0) and other columns
        self.tree = ttk.Treeview(table_frame, columns=("PayAddress", "Amount"), show="tree headings", height=5)

        # Configure columns
        self.tree.heading("#0", text="Image")
        self.tree.column("#0", width=150)

        self.tree.heading("PayAddress", text="PayAddress")
        self.tree.column("PayAddress", width=200)

        self.tree.heading("Amount", text="Amount")
        self.tree.column("Amount", width=100)

        self.tree.pack(fill='both', expand=True)

        # # Load your image
        # pil_image = Image.open(r"C:\Users\beast\Downloads\bf652f6d-51c1-4b6e-b12c-81fab4000a3b\output_image_1.png").resize((180, 180), Image.ANTIALIAS)
        # my_image_tk = ImageTk.PhotoImage(pil_image)

        # # Keep a reference to avoid garbage collection
        self.stored_images = []
        # self.stored_images.append(my_image_tk)

        # # Insert data with image in the #0 column
        # self.tree.insert("", "end", text="", image=my_image_tk, values=("bc1pn2yh86uw8uqde3kn7yjlfnmx97qgzl8vzmjulhxd2820smaat2jswzd662", "134241"))

        # Bind a double-click event to initiate copy
        self.tree.bind("<Double-1>", self.on_double_click)

        # For demonstration, store references to images loaded in table (optional)
        # self.table_images = []

        # TODO: Add any additional UI elements or placeholders as required

    def on_double_click(self, event):
        """Handle double-click to copy cell's value."""
        # Identify the item and column clicked
        region = self.tree.identify("region", event.x, event.y)

        if region == "cell":
            column_id = self.tree.identify_column(event.x)
            row_id = self.tree.identify_row(event.y)

            if row_id and column_id:
                # Get column index (omit the '#' from column_id)
                col_index = int(column_id.replace('#', '')) - 1  # -1 because columns start at #1
                # Get values from the row
                row_values = self.tree.item(row_id, "values")

                # If the clicked column is the tree column (#0), we need a different approach:
                # For the tree column, the displayed text is item(row_id, "text")
                # For headings columns:
                if column_id == "#0":
                    cell_value = self.tree.item(row_id, "text")
                else:
                    # Extract the specific cell's value from the tuple
                    cell_value = row_values[col_index] if col_index < len(row_values) else ""

                # Copy to clipboard
                self.clipboard_clear()
                self.clipboard_append(cell_value)

                # Optionally, show a notification or status
                print(f"Copied: {cell_value}")

    def select_image(self):
        """Prompt user to select an image file."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            try:
                img = Image.open(file_path).resize((256, 256), Resampling.LANCZOS)
                self.selected_image = ImageTk.PhotoImage(img)
                self.img_label.config(image=self.selected_image, text="")
                self.selected_image_path = file_path
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{e}")
                self.img_label.config(text="No image selected", image='')
                self.selected_image = None

    def clear_fields(self):
        """Clear the input fields and reset the image."""
        self.rec_addr_var.set("")
        self.nfts_var.set("1")
        self.prompt_text.delete("1.0", tk.END)
        self.img_label.config(text="No image selected", image='')
        self.selected_image = None
        self.selected_image_path = None
        self.nft_responses = []
        self.tree.delete(*self.tree.get_children())
        # TODO: Clear table if needed

    def generate_nfts(self):
        """Start the NFT generation process."""
        self.generate_btn.config(state='disabled', text="Generating...")

        # Start a thread to simulate background process, replace with actual logic
        threading.Thread(target=self.generate_nfts_logic).start()

    def generate_nfts_logic(self):
        """Placeholder for the actual NFT generation logic."""
        try:
            output_dir = create_output_directory()
            for i in range(int(self.nfts_var.get())):
                source_path = self.selected_image_path
                target_path = generate_target_image(output_dir, self.prompt_text.get("1.0", tk.END), source_path, i)

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
                    "receiveAddress": self.rec_addr_var.get(),  # Replace with actual address
                    "feeRate": 1,
                    "outputValue": 546,
                    "files": [
                        {
                            "filename": os.path.basename(output_path),  # Dynamically set filename
                            "dataURL": data_url  # Use the generated Data URL
                        }
                    ]
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

                    nft_response = response.json()

                    data.append(nft_response)
                    json.dump(data, open("nft_response.json", "w"))
                    # self.nft_responses.append({
                    #     "image": output_path,
                    #     "nft_detail": response.json(),
                    # })

                    pil_image = Image.open(output_path).resize((180, 180), Resampling.LANCZOS)
                    my_image_tk = ImageTk.PhotoImage(pil_image)
                    self.stored_images.append(my_image_tk)
                    # Insert data with image in the #0 column
                    self.tree.insert("", "end", text="", image=my_image_tk,
                                     values=(nft_response["data"]["payAddress"],
                                             f"0.00{nft_response['data']['amount']}"))
                except requests.exceptions.RequestException as e:
                    print(f"Failed to create NFT for Image {i + 1}: {e}")

                print(f"Image {i + 1} generated successfully: {output_path}, proceeding to next image (if requested)...")

            # If success:
            self.after(0, self.on_generation_complete)
        except Exception as e:
            # If error occurs
            print(f"An error occurred: {e}")
            self.after(0, lambda ex=e: self.on_generation_error(ex))

    def on_generation_complete(self):
        """Handle UI update after successful NFT generation."""
        self.generate_btn.config(state='normal', text="Generate NFTs")
        messagebox.showinfo("Success", "NFTs generated successfully!")

        # for nft_response in self.nft_responses:
        #     # Load your image
        #     pil_image = Image.open(nft_response["image"]).resize((180, 180), Resampling.LANCZOS)
        #     my_image_tk = ImageTk.PhotoImage(pil_image)
        #     self.stored_images.append(my_image_tk)
        #     # Insert data with image in the #0 column
        #     self.tree.insert("", "end", text="", image=my_image_tk,
        #                      values=(nft_response["nft_detail"]["data"]["payAddress"],
        #                              f"0.00{nft_response['nft_detail']['data']['amount']}"))
        return None

    def on_generation_error(self, err):
        """Handle UI update and show error if NFT generation fails."""
        self.generate_btn.config(state='normal', text="Generate NFTs")
        messagebox.showerror("Error", f"An error occurred:\n{err}")
        # TODO: Additional error handling if needed
        return None


if __name__ == "__main__":
    app = NFTApp()
    app.mainloop()
