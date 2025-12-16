import torch
import re
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from PIL import Image


class SuryaOCR:
    def __init__(self):
        # 1. GPU Check
        # Surya automatically uses CUDA if available, but we log it for clarity.
        if torch.cuda.is_available():
            print(f"✅ Surya OCR: GPU Detected ({torch.cuda.get_device_name(0)})")
            self.device = "cuda"
        else:
            print("⚠️ Surya OCR: GPU NOT found. Falling back to CPU (Slow).")
            self.device = "cpu"

        print("👁️ Loading Surya Models...")
        try:
            # 2. Load Models
            # Foundation model is required for accurate layout analysis in v0.17+
            self.foundation = FoundationPredictor()
            self.rec_predictor = RecognitionPredictor(self.foundation)
            self.det_predictor = DetectionPredictor()
            print("✅ Surya Models Loaded Successfully.")
        except Exception as e:
            print(f"❌ Error loading Surya Models: {e}")
            raise e

    def is_valid_line(self, text):
        """
        Filters out common OCR noise and hallucinations.
        """
        if not text: return False

        text = text.strip()
        if len(text) < 2: return False  # Skip single random chars

        # Filter Chinese/Asian characters which often appear as hallucinations
        # in noisy receipt backgrounds (unless you specifically expect them).
        if re.search(r'[\u4e00-\u9fff]', text):
            return False

        return True

    def run(self, image_pil: Image.Image) -> str:
        """
        Takes a PIL Image -> Returns Raw Text String
        """
        try:
            # Run Detection + Recognition
            # We pass [None] as the second argument because we don't have language hints
            predictions = self.rec_predictor([image_pil], [None], det_predictor=self.det_predictor)

            result = predictions[0]

            # Extract and filter text lines
            lines = []
            for line in result.text_lines:
                if self.is_valid_line(line.text):
                    lines.append(line.text)

            # Join with newlines to preserve receipt structure
            return "\n".join(lines)

        except Exception as e:
            print(f"❌ Surya OCR Failed: {e}")
            return ""