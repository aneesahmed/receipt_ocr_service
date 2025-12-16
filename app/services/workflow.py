from app.processors.cropper import ImageCropper
from app.processors.ocr_engine import SuryaOCR
from app.processors.parser import ReceiptParser
from PIL import Image
import cv2

# Initialize Singletons (Loads models once at startup)
cropper = ImageCropper()
ocr = SuryaOCR()
parser = ReceiptParser()


def pipeline_process_image(image_bytes: bytes) -> dict:
    # 1. Crop
    cropped_cv2 = cropper.process(image_bytes)
    if cropped_cv2 is None:
        return {"error": "Cropping failed or image empty"}

    # Convert OpenCV (BGR) to PIL (RGB)
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_cv2, cv2.COLOR_BGR2RGB))

    # 2. OCR
    raw_text = ocr.run(cropped_pil)

    # 3. Parse
    json_data = parser.parse(raw_text)

    return json_data