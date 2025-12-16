from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import logging

# Import the workflow functions that handle the heavy lifting
from app.services.workflow import (
    workflow_vision_direct,
    workflow_surya_pipeline
)

# 1. Logging Configuration
# This ensures you see "Vision Request received" in your Docker logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("receipt-api")

# 2. Initialize App
app = FastAPI(
    title="Receipt OCR Service",
    description="Dual-Pipeline OCR Service: Qwen-Vision (Visual) vs Surya+Qwen (Text Logic)",
    version="1.0.0"
)


@app.get("/")
def health_check():
    """Simple check to see if the server is running."""
    return {
        "status": "online",
        "endpoints": [
            "POST /ocr/vision - Best for complex layouts & tables",
            "POST /ocr/surya  - Best for gas stations & faint text"
        ]
    }


# =========================================================
#  ENDPOINT 1: VISION MODEL (Direct Image -> JSON)
# =========================================================
@app.post("/ocr/vision")
async def endpoint_vision(file: UploadFile = File(...)):
    """
    Pipeline A:
    1. Receives Raw Image
    2. Crops it (GPU) internaly
    3. Sends crop directly to Qwen-VL (Vision Model)
    4. Returns JSON
    """
    try:
        logger.info(f"👁️ Vision Request received: {file.filename}")

        # Read file into memory (bytes)
        data = await file.read()

        # Execute Workflow
        result = workflow_vision_direct(data)

        # Check for specific workflow errors
        if "error" in result:
            logger.warning(f"Vision failed: {result['error']}")
            return JSONResponse(content=result, status_code=400)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"🔥 Critical Vision Error: {str(e)}")
        return JSONResponse(content={"error": "Internal Server Error", "details": str(e)}, status_code=500)


# =========================================================
#  ENDPOINT 2: SURYA PIPELINE (OCR -> Text Parsing)
# =========================================================
@app.post("/ocr/surya")
async def endpoint_surya(file: UploadFile = File(...)):
    """
    Pipeline B:
    1. Receives Raw Image
    2. Crops it (GPU) internally
    3. Extracts Raw Text (Surya OCR)
    4. Parses Text into JSON (Qwen Text Model)
    5. Returns JSON
    """
    try:
        logger.info(f"🧠 Surya Request received: {file.filename}")

        # Read file into memory (bytes)
        data = await file.read()

        # Execute Workflow
        result = workflow_surya_pipeline(data)

        if "error" in result:
            logger.warning(f"Surya failed: {result['error']}")
            return JSONResponse(content=result, status_code=400)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"🔥 Critical Surya Error: {str(e)}")
        return JSONResponse(content={"error": "Internal Server Error", "details": str(e)}, status_code=500)