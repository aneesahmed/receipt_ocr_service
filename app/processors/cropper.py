import cv2
import numpy as np
import onnxruntime as ort
from rembg import remove, new_session
from PIL import Image


class ImageCropper:
    def __init__(self, model_name="birefnet-general"):
        # 1. GPU Check for Rembg/ONNX
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print(f"✅ Cropper: GPU Enabled (CUDA Found)")
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            print(f"⚠️ Cropper: GPU NOT found. Using CPU.")
            self.providers = ['CPUExecutionProvider']

        print(f"✂️ Loading Rembg Model: {model_name}...")
        self.session = new_session(model_name, providers=self.providers)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # TL
        rect[2] = pts[np.argmax(s)]  # BR
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def process(self, image_bytes: bytes) -> np.ndarray:
        # 1. Background Removal
        mask_bytes = remove(image_bytes, session=self.session, only_mask=True)
        mask_np = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)

        # 2. Decode Original
        nparr = np.frombuffer(image_bytes, np.uint8)
        original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original is None: return None

        # 3. Find Receipt Contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return original  # Return original if no object found

        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 4. Warp (Scanner Effect) or Deskew (Crop)
        if len(approx) == 4:
            processed = self.four_point_transform(original, approx.reshape(4, 2))
        else:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            processed = self.four_point_transform(original, box)

        # 5. Enforce Portrait Orientation
        h, w = processed.shape[:2]
        if w > h:
            processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)

        return processed