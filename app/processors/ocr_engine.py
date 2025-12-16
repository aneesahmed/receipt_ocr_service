import torch
import re
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from PIL import Image


class SuryaOCR:
    def __init__(self):
        # 1. GPU Check
        if torch.cuda.is_available():
            print(f"✅ OCR Engine: GPU Detected ({torch.cuda.get_device_name(0)})")
        else:
            print("⚠️ OCR Engine: GPU NOT found. Falling back to CPU (Slow).")

        print("👁️ Loading Surya Models...")
        try:
            # Load foundation first (Required for v0.17+)
            self.foundation = FoundationPredictor()
            self.rec_predictor = RecognitionPredictor(self.foundation)
            self.det_predictor = DetectionPredictor()
            print("✅ Surya Models Loaded.")
        except Exception as e:
            print(f"❌ Error loading Surya: {e}")
            raise e

    def is_valid_line(self, text):
        if not text or len(text.strip()) == 0: return False
        # Filter Chinese/Asian characters (common hallucination in noise)
        if re.search(r'[\u4e00-\u9fff]', text): return False
        return True

    def run(self, image_pil: Image.Image) -> str:
        # Run OCR with detection + recognition
        predictions = self.rec_predictor([image_pil], [None], det_predictor=self.det_predictor)
        result = predictions[0]

        lines = []
        for line in result.text_lines:
            if self.is_valid_line(line.text):
                lines.append(line.text)

        return "\n".join(lines)