from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import JSONResponse
import zipfile
import io
import json
import logging
from app.services.workflow import pipeline_process_image

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Receipt OCR Service")


@app.get("/")
def health_check():
    return {"status": "active", "service": "Receipt OCR"}


@app.post("/process/image")
async def process_single(file: UploadFile = File(...)):
    """Upload 1 image -> Get JSON"""
    try:
        logger.info(f"Processing single file: {file.filename}")
        image_bytes = await file.read()
        result = pipeline_process_image(image_bytes)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/process/batch")
async def process_batch(file: UploadFile = File(...)):
    """Upload ZIP -> Get ZIP with JSONs"""
    try:
        logger.info(f"Processing batch zip: {file.filename}")
        zip_bytes = await file.read()
        input_zip = zipfile.ZipFile(io.BytesIO(zip_bytes))
        output_io = io.BytesIO()

        processed_count = 0

        with zipfile.ZipFile(output_io, 'w', zipfile.ZIP_DEFLATED) as out_zip:
            for filename in input_zip.namelist():
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    try:
                        with input_zip.open(filename) as img_file:
                            img_data = img_file.read()

                        # Run Pipeline
                        result = pipeline_process_image(img_data)

                        # Save JSON
                        out_zip.writestr(f"{filename}.json", json.dumps(result, indent=2))
                        processed_count += 1
                        logger.info(f"Processed: {filename}")
                    except Exception as e:
                        out_zip.writestr(f"{filename}_error.txt", str(e))

        output_io.seek(0)
        logger.info(f"Batch complete. Processed {processed_count} files.")

        return Response(
            content=output_io.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=results.zip"}
        )

    except Exception as e:
        logger.error(f"Batch Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)