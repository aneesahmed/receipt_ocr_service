import io
import cv2
from PIL import Image
# Import your 4 processors
from app.processors.cropper import ImageCropper
from app.processors.surya_ocr import SuryaOCR
from app.processors.surya_ocr_parser import SuryaParser
from app.processors.ollama_vision_ocr import OllamaVisionOCR

# ==========================================
# 1. INITIALIZE SINGLETONS (Run once at startup)
# ==========================================
print("🚀 Initializing Workflow Services...")
cropper = ImageCropper()
surya_engine = SuryaOCR()
surya_parser = SuryaParser()
vision_engine = OllamaVisionOCR()
print("✅ Workflow Services Ready.")


# ==========================================
# 2. HELPER: Pre-processing
# ==========================================
def _crop_and_prep(image_bytes: bytes):
    """
    Shared Logic: 
    1. Send raw bytes to GPU Cropper.
    2. Convert result to PIL (for Surya).
    3. Convert result to PNG Bytes (for Vision).
    """
    cropped_cv2 = cropper.process(image_bytes)

    if cropped_cv2 is None:
        return None, None

    # Convert OpenCV (BGR) -> PIL (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(cropped_cv2, cv2.COLOR_BGR2RGB))

    # Convert PIL -> Bytes (PNG)
    output_io = io.BytesIO()
    image_pil.save(output_io, format='PNG')
    cropped_bytes = output_io.getvalue()

    return image_pil, cropped_bytes


# ==========================================
# 3. PIPELINE A: VISION DIRECT
# ==========================================
def workflow_vision_direct(image_bytes: bytes) -> dict:
    # Step 1: Crop
    _, cropped_bytes = _crop_and_prep(image_bytes)
    if not cropped_bytes:
        return {"error": "Cropping failed - could not detect receipt"}

    # Step 2: Vision Model
    return vision_engine.parse(cropped_bytes)


# ==========================================
# 4. PIPELINE B: SURYA + TEXT PARSER
# ==========================================
def workflow_surya_pipeline(image_bytes: bytes) -> dict:
    # Step 1: Crop
    image_pil, _ = _crop_and_prep(image_bytes)
    if not image_pil:
        return {"error": "Cropping failed - could not detect receipt"}

    # Step 2: Extract Text (Surya)
    raw_text = surya_engine.run(image_pil)

    # Step 3: Parse Text (Qwen Text Model)
    return surya_parser.parse(raw_text)


# ... (Previous imports and init code) ...

# ==========================================
# 5. PIPELINE C: CROP ONLY (Returns Bytes)
# ==========================================
def workflow_get_cropped_image(image_bytes: bytes):
    """
    Returns the raw PNG bytes of the cropped receipt.
    """
    # Use the shared helper we already wrote
    _, cropped_bytes = _crop_and_prep(image_bytes)

    if not cropped_bytes:
        return None

    return cropped_bytes