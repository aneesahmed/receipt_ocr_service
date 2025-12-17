import cv2
import numpy as np
import onnxruntime as ort
from rembg import remove, new_session


class ImageCropper:
    def __init__(self, model_name="birefnet-general"):
        # Check for GPU
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"✅ Cropper: GPU Enabled.")
        else:
            self.providers = ['CPUExecutionProvider']
            print(f"⚠️ Cropper: GPU NOT found. Using CPU.")

        self.session = new_session(model_name, providers=self.providers)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
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
        # 1. Decode Original
        nparr = np.frombuffer(image_bytes, np.uint8)
        original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original is None: return None

        # 2. Remove Background (Get Mask)
        # using alpha_matting=True helps with edges on dark backgrounds
        mask_bytes = remove(image_bytes, session=self.session, only_mask=True, alpha_matting=True)
        mask_np = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)

        # 3. Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return original  # Fallback to original if Rembg fails completely

        # Get largest contour (the receipt)
        c = max(contours, key=cv2.contourArea)

        # 4. SAFETY CHECK: Is the contour big enough?
        # If it's too small (noise), return original
        if cv2.contourArea(c) < 5000:
            return original

        # 5. NEW LOGIC: "Safe Crop" (Bounding Box) vs "Warp"
        # We prefer a simple bounding box crop for curved receipts
        # because warping creates distortion.
        x, y, w, h = cv2.boundingRect(c)

        # Add a small padding (10px) to not cut edge text
        h_img, w_img = original.shape[:2]
        x = max(0, x - 10)
        y = max(0, y - 10)
        w = min(w_img - x, w + 20)
        h = min(h_img - y, h + 20)

        cropped = original[y:y + h, x:x + w]

        # 6. Rotate if Landscape (Make it tall)
        h_c, w_c = cropped.shape[:2]
        if w_c > h_c:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

        return cropped