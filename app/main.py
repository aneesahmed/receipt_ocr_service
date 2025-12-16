from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import JSONResponse
import logging

# 1. Import Workflow Functions
from app.services.workflow import (
    workflow_vision_direct,
    workflow_surya_pipeline,
    workflow_get_cropped_image
)

# 2. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("receipt-api")

# 3. INITIALIZE APP (This must happen BEFORE @app.post)
app = FastAPI(
    title="Receipt OCR Service",
    description="Dual-Pipeline OCR Service: Qwen-Vision vs Surya+Qwen",
    version="1.0.0"
)


# 4. Define Endpoints
@app.get("/")
def health_check():
    return {"status": "online", "endpoints": ["/ocr/vision", "/ocr/surya", "/process/crop"]}


# --- ENDPOINT 1: VISION MODEL ---
@app.post("/ocr/vision")
async def endpoint_vision(file: UploadFile = File(...)):
    try:
        logger.info(f"👁️ Vision Request: {file.filename}")
        data = await file.read()
        result = workflow_vision_direct(data)

        if "error" in result:
            return JSONResponse(result, status_code=400)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Vision Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# --- ENDPOINT 2: SURYA PIPELINE ---
@app.post("/ocr/surya")
async def endpoint_surya(file: UploadFile = File(...)):
    try:
        logger.info(f"🧠 Surya Request: {file.filename}")
        data = await file.read()
        result = workflow_surya_pipeline(data)

        if "error" in result:
            return JSONResponse(result, status_code=400)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Surya Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# --- ENDPOINT 3: CROP PREVIEW ---
@app.post("/process/crop")
async def endpoint_crop_preview(file: UploadFile = File(...)):
    try:
        logger.info(f"✂️ Crop Request: {file.filename}")
        data = await file.read()
        cropped_bytes = workflow_get_cropped_image(data)

        if not cropped_bytes:
            return JSONResponse({"error": "Crop failed"}, status_code=400)

        # Return actual image
        return Response(content=cropped_bytes, media_type="image/png")
    except Exception as e:
        logger.error(f"Crop Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)