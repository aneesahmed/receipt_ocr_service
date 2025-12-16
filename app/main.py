from fastapi import FastAPI, UploadFile, File, Response  # Added Response
from fastapi.responses import JSONResponse
import logging

# Import the new function
from app.services.workflow import (
    workflow_vision_direct,
    workflow_surya_pipeline,
    workflow_get_cropped_image  # <--- Import this
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ... (Previous Setup Code) ...

# =========================================================
#  ENDPOINT 3: GET CROPPED IMAGE
# =========================================================
@app.post("/process/crop")
async def endpoint_crop_preview(file: UploadFile = File(...)):
    """
    1. Receives Raw Image
    2. Crops it (GPU)
    3. Returns the actual PNG Image (not JSON)
    """
    try:
        logger.info(f"✂️ Crop Request received: {file.filename}")
        data = await file.read()

        # Get raw bytes
        cropped_bytes = workflow_get_cropped_image(data)

        if not cropped_bytes:
            return JSONResponse(
                {"error": "Could not detect receipt to crop"},
                status_code=400
            )

        # Return as an Image File
        return Response(content=cropped_bytes, media_type="image/png")

    except Exception as e:
        logger.error(f"Crop Error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)