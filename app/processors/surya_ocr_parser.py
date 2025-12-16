from ollama import Client
import json
import re

# --- V6 SYSTEM PROMPT (Optimized for Text Input) ---
SYSTEM_PROMPT = """
You are a specialized Receipt OCR Engine. Convert the raw text into strict JSON.

### 1. STORE NAME STRATEGY (CRITICAL)
- **Ignore Generic Headers:** If the first line is "Transaction Record", "Merchant Copy", "Original", or "Welcome", **skip it**.
- **Look Deeper:** The store name is usually the **first distinct business name** in the top 5 lines.
- **Logo Text:** If the OCR text is messy at the top, look for the largest brand name text.

### 2. GAS/FUEL STATION LOGIC
- **Item Extraction:** If you see "Pump", "Regular", "Diesel", or "Fuel Sales":
  - **Description:** Use the fuel grade (e.g., "Regular").
  - **Quantity:** Extract the volume in Liters (e.g., "42.619L" -> 42.619).
  - **Price:** Use the **Unit Price** (e.g., "$1.619/L" -> 1.619).
  - **Consistency:** `qty * price` must roughly equal the line total.

### 3. GENERAL ITEMS
- **Exclude Financials:** Do NOT list "Subtotal", "Tax", "Total", "Balance Due", or "Payment" as line items.
- **Extract Products:** Only extract distinct products/services.

### 4. DATE FIX
- **Assume Current Era:** 2024-2025.
- If you see "23" or "25" but the date is ambiguous, prefer 2024 unless clearly stated otherwise.
- Format: YYYY-MM-DD.

### JSON OUTPUT FORMAT
{
    "store_name": "string (or null)",
    "date": "YYYY-MM-DD (or null)",
    "time": "HH:MM (or null)",
    "total_amount": 0.00,
    "taxes": { "tps": 0.00, "tvq": 0.00 },
    "items": [
        { "qty": 1.0, "desc": "string", "price": 0.00 }
    ]
}
"""

class SuryaParser:
    def __init__(self, host="http://173.209.56.38:11434"):
        self.client = Client(host=host)
        # Using the standard Text model for parsing text input
        self.model = "qwen2.5:7b-instruct-q4_K_M"
        print(f"🧠 Text Parser: Connected to Ollama ({self.model})")

    def extract_json(self, text):
        """
        Robustly finds the first { and the last } to extract JSON
        ignoring all intro/outro text.
        """
        try:
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return None
        except:
            return None

    def parse(self, text_content: str) -> dict:
        """
        Pipeline: Raw Text -> LLM (Attempt 1) -> LLM Repair (Attempt 2) -> JSON
        """
        if not text_content or len(text_content.strip()) < 5:
            return {"error": "OCR Text is empty or too short"}

        try:
            # --- ATTEMPT 1: Main Extraction ---
            response = self.client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': f"RAW TEXT:\n{text_content}"}
                ]
            )
            content = response['message']['content']
            parsed = self.extract_json(content)

            # --- ATTEMPT 2: Repair if Failed ---
            if not parsed:
                print("⚠️ Text Parser: JSON invalid. Attempting repair...")
                repair_resp = self.client.chat(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': "You are a code fixer. Fix the following invalid JSON. Remove any math expressions (e.g. '5*2' -> '10'). Return ONLY JSON."},
                        {'role': 'user', 'content': content}
                    ]
                )
                parsed = self.extract_json(repair_resp['message']['content'])

            if parsed:
                return parsed
            else:
                return {
                    "error": "Parsing Failed",
                    "raw_response": content
                }

        except Exception as e:
            return {"error": f"Ollama Error: {str(e)}"}