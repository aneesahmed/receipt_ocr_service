from ollama import Client
import json
import re

# Strict Prompt for Receipt parsing
SYSTEM_PROMPT = """
You are a specialized Receipt OCR Engine. Convert the raw text into strict JSON.
Ignore generic headers. Extract store name, date, total, and line items.
Format:
{
    "store_name": "string",
    "date": "YYYY-MM-DD",
    "total_amount": 0.00,
    "items": [{"desc": "string", "price": 0.00, "qty": 1}]
}
Return ONLY valid JSON.
"""


class ReceiptParser:
    def __init__(self, host="http://173.209.56.38:11434", model="qwen2.5:7b-instruct-q4_K_M"):
        self.client = Client(host=host)
        self.model = model
        print(f"🤖 Parser connected to Ollama at {host}")

    def parse(self, ocr_text: str) -> dict:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': f"RAW TEXT:\n{ocr_text}"}
                ]
            )
            content = response['message']['content']

            # Robust JSON extraction
            match = re.search(r'(\{.*\})', content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return {"error": "JSON Parsing Failed", "raw": content}
        except Exception as e:
            return {"error": f"Ollama Error: {str(e)}"}