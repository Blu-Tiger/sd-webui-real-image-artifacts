import random
import os
import io
import logging
import piexif
import gradio as gr
from PIL import Image, ImageEnhance
import numpy as np
from modules import scripts_postprocessing, shared, ui_components
import modules


def search_extras_folder(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if 'output' in dirs or 'outputs' in dirs:
            output_folder = os.path.join(
                root, 'output') if 'output' in dirs else os.path.join(root, 'outputs')
            for root, dirs, files in os.walk(output_folder):
                for dir_name in dirs:
                    if 'extra' in dir_name:
                        extra_folder_path = os.path.join(
                            output_folder, dir_name)
                        logging.debug("Real Image Artifact==> Found extra images folder at: %s",
                                      extra_folder_path)
                        return extra_folder_path
            logging.error(
                "Real Image Artifact==> extras or extras-images folder is missing")
        else:
            logging.error("Real Image Artifact==> Output folder is missing")


lens_make_model_list = [
    ("Canon", "EF 24-70mm f/2.8L II USM"),
    ("Nikon", "AF-S NIKKOR 50mm f/1.8G"),
    ("Sony", "FE 24-70mm f/2.8 GM"),
    ("Sigma", "35mm f/1.4 DG HSM Art"),
    ("Tamron", "SP 70-200mm f/2.8 Di VC USD G2"),
    ("Leica", "Summilux-M 35mm f/1.4 ASPH"),
    ("Fujifilm", "XF 16-55mm f/2.8 R LM WR"),
    ("Panasonic", "Lumix S Pro 50mm f/1.4"),
    ("Zeiss", "Otus 55mm f/1.4"),
    ("Olympus", "M.Zuiko Digital ED 12-40mm f/2.8 PRO"),
    ("Pentax", "HD DA 20-40mm f/2.8-4 Limited DC WR"),
    ("Samsung", "NX 16-50mm f/2-2.8 S ED OIS"),
    ("Tokina", "AT-X 11-20mm f/2.8 PRO DX"),
    ("Voigtländer", "Nokton 40mm f/1.2 Aspherical"),
    ("Yongnuo", "YN 50mm f/1.8 II"),
    ("Hasselblad", "XCD 45mm f/3.5"),
    ("Rokinon", "SP 14mm f/2.4"),
    ("Samyang", "AF 85mm f/1.4 EF"),
    ("Tokina", "Opera 16-28mm f/2.8 FF"),
    ("Zeiss", "Batis 85mm f/1.8")
]

camera_owner_list = [
    "Alice",
    "Bob",
    "Catherine",
    "David",
    "Elena",
    "Frank",
    "Grace",
    "Melissa",
    "Isabel",
    "Jack"
]

def_metadata_settings = {
    "LensMake": "Canon",
    "LensModel": "EF 24-70mm f/2.8L II USM",
    "CameraOwnerName": "Melissa",
    "BodySerialNumber": str(random.randint(1000000, 9999999)),
    "LensSerialNumber": str(random.randint(1000000, 9999999)),
    "FocalLength": f"{random.randint(24, 70)},1",
    "FNumber": f"{random.randint(28, 56)},10",
    "ExposureTime": f"{random.randint(1, 1000)},1000",
    "ISOSpeedRatings": str(random.randint(100, 6400))
}


def randomize_metadata():
    lens_make, lens_model = random.choice(lens_make_model_list)
    rabdom_data = {
        "LensMake": lens_make,
        "LensModel": lens_model,
        "CameraOwnerName": random.choice(camera_owner_list),
        "BodySerialNumber": str(random.randint(1000000, 9999999)),
        "LensSerialNumber": str(random.randint(1000000, 9999999)),
        "FocalLength": f"{random.randint(24, 70)},1",
        "FNumber": f"{random.randint(28, 56)},10",
        "ExposureTime": f"{random.randint(1, 1000)},1000",
        "ISOSpeedRatings": str(random.randint(100, 6400))
    }
    return [value for key, value in rabdom_data.items()]


def random_exif_data(metadata_settings):
    exif_ifd = {
        piexif.ExifIFD.LensMake: metadata_settings["LensMake"],
        piexif.ExifIFD.LensModel: metadata_settings["LensModel"],
        piexif.ExifIFD.CameraOwnerName: metadata_settings["CameraOwnerName"],
        piexif.ExifIFD.BodySerialNumber: metadata_settings["BodySerialNumber"],
        piexif.ExifIFD.LensSerialNumber: metadata_settings["LensSerialNumber"],
        piexif.ExifIFD.FocalLength: tuple(map(int, metadata_settings["FocalLength"].split(','))),
        piexif.ExifIFD.FNumber: tuple(map(int, metadata_settings["FNumber"].split(','))),
        piexif.ExifIFD.ExposureTime: tuple(map(int, metadata_settings["ExposureTime"].split(','))),
        piexif.ExifIFD.ISOSpeedRatings: int(
            metadata_settings["ISOSpeedRatings"])
    }
    exif_dict = {"Exif": exif_ifd}
    exif_bytes = piexif.dump(exif_dict)
    return exif_bytes


def add_realistic_noise(img, base_noise_level):
    img_array = np.array(img)

    # Base noise: Gaussian noise
    noise = np.random.normal(0, 255 * base_noise_level, img_array.shape)

    # Noise variation: Some pixels have stronger noise
    noise_variation = np.random.normal(
        0, 255 * (base_noise_level * 2), img_array.shape)
    variation_mask = np.random.rand(
        *img_array.shape[:2]) > 0.95  # Mask for stronger noise
    noise[variation_mask] += noise_variation[variation_mask]

    # Applying the noise to the image
    noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_img_array)


def convert_to_rgb(img):
    # Convert from RGBA to RGB if necessary
    if img.mode == 'RGBA':
        return img.convert('RGB')
    return img


def worst_image(img, noise_level, jpeg_artifact_level, enable_exif, metadata_settings):
    try:
        output_buffer = io.BytesIO()

        # Applying image degradation processes
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))

        img = add_realistic_noise(
            img, base_noise_level=noise_level)

        img.save(output_buffer, 'JPEG', quality=jpeg_artifact_level)

        try:
            if enable_exif:
                folder_path = search_extras_folder(modules.scripts.basedir())

                file_names = [f for f in os.listdir(
                    folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                numbers = [int(''.join(filter(str.isdigit, name)))
                           for name in file_names if any(char.isdigit() for char in name)]
                if numbers:
                    max_number = max(numbers)
                    new_file_name = f"{max_number + 1:05d}_exif.jpg"
                else:
                    new_file_name = f"{0:05d}_exif.jpg"

                new_file_path = os.path.join(folder_path, new_file_name)

                with Image.open(output_buffer) as sv_img:
                    sv_img.save(new_file_path, exif=random_exif_data(
                        metadata_settings))
                    logging.info("Real Image Artifact==> Exif image saved: %s",
                                 new_file_path)
        except Exception as e:
            logging.error("Real Image Artifact==> Error exif: %s", e)

        final_img = Image.open(output_buffer)
        return final_img
    except Exception as e:
        logging.error("Real Image Artifact==> Error processing: %s", e)


class ScriptPostprocessingRealImageArtifact(scripts_postprocessing.ScriptPostprocessing):
    name = "Real Image Artifact"
    order = 5000

    def ui(self):
        with ui_components.InputAccordion(False, label="Real Image Artifact") as enable:
            with ui_components.InputAccordion(False, label="Exif Generator") as enable_exif:
                rad_metadata = ui_components.ToolButton(value='🔄️')
                LensMake = gr.Textbox(label='Camera lens maker',
                                      value=def_metadata_settings["LensMake"])
                LensModel = gr.Textbox(
                    label='Camera lens model', value=def_metadata_settings["LensModel"])
                CameraOwnerName = gr.Textbox(label='Camera owner name',
                                             value=def_metadata_settings["CameraOwnerName"])
                BodySerialNumber = gr.Textbox(label='Camera body serial number',
                                              value=def_metadata_settings["BodySerialNumber"])
                LensSerialNumber = gr.Textbox(label='Camera lens serial number',
                                              value=def_metadata_settings["LensSerialNumber"])
                FocalLength = gr.Textbox(label='Focal length',
                                         value=def_metadata_settings["FocalLength"])
                FNumber = gr.Textbox(label='FNumber',
                                     value=def_metadata_settings["FNumber"])
                ExposureTime = gr.Textbox(label='Exposure time',
                                          value=def_metadata_settings["ExposureTime"])
                ISOSpeedRatings = gr.Textbox(label='ISO speed ratings',
                                             value=def_metadata_settings["ISOSpeedRatings"])
            with gr.Group():
                noise_level = gr.Slider(minimum=0, maximum=1, label='Noise level', step=0.01, info="Higer for worse quality",
                                        value=0.03)
                jpeg_artifact_level = gr.Slider(minimum=0, maximum=100, label='Jpeg quality', step=1, info="Lower for worse quality",
                                                value=50)
        rad_metadata.click(fn=randomize_metadata, outputs=[
                           LensMake, LensModel,  CameraOwnerName, BodySerialNumber, LensSerialNumber, FocalLength, FNumber, ExposureTime, ISOSpeedRatings])

        return {
            "noise_level": noise_level,
            "jpeg_artifact_level": jpeg_artifact_level,
            "enable_exif": enable_exif,
            "LensMake": LensMake,
            "LensModel": LensModel,
            "CameraOwnerName": CameraOwnerName,
            "BodySerialNumber": BodySerialNumber,
            "LensSerialNumber": LensSerialNumber,
            "FocalLength": FocalLength,
            "FNumber": FNumber,
            "ExposureTime": ExposureTime,
            "ISOSpeedRatings": ISOSpeedRatings
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, noise_level, jpeg_artifact_level, enable_exif,  LensMake, LensModel,  CameraOwnerName, BodySerialNumber, LensSerialNumber, FocalLength, FNumber, ExposureTime, ISOSpeedRatings):
        metadata_settings = {
            "LensMake": LensMake,
            "LensModel": LensModel,
            "CameraOwnerName": CameraOwnerName,
            "BodySerialNumber": BodySerialNumber,
            "LensSerialNumber": LensSerialNumber,
            "FocalLength": FocalLength,
            "FNumber": FNumber,
            "ExposureTime": ExposureTime,
            "ISOSpeedRatings": ISOSpeedRatings
        }
        pp.image = worst_image(pp.image.convert(
            "RGB"),  noise_level, jpeg_artifact_level, enable_exif, metadata_settings)
