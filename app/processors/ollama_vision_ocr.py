from ollama import Client
import json
import re

# --- MERGED PROMPT: VISION CAPABILITIES + BUSINESS LOGIC ---
VISION_PROMPT = """
You are a specialized receipt OCR engine. Extract all data into valid JSON.

### 1. STORE NAME STRATEGY
- **Ignore Generic Headers:** If the top text is "Transaction Record", "Merchant Copy", "Original", or "Welcome", skip it.
- **Find the Brand:** The store name is usually the largest text or the first distinct business name in the top 20% of the image.

### 2. LINE ITEM LOGIC (CRITICAL)
- **Read Every Line:** Do not skip items. If an item appears multiple times, list it multiple times.
- **Exclude Totals:** Do NOT include "Subtotal", "Tax", "Total", "Balance", or "Change" inside the `items` list.
- **Price extraction:**
  - **Standard Items:** Use the **Final Line Total** for `price`. Set `qty` to 1 unless distinct quantity is visible.
  - **Gas/Fuel (Special Case):** If you see "Pump", "Regular", "Diesel", or "Fuel":
    - Set `desc` = Fuel Grade (e.g. "Regular").
    - Set `qty` = Volume in Liters (e.g. 42.619).
    - Set `price` = Unit Price per Liter (e.g. 1.659).

### 3. TAX & FINANCIALS
- **Tax Amounts:** Look for monetary values labeled "TPS", "TVQ", "GST", "HST", "QST". Extract the **Amount** ($), not the registration number.
- **Date Fix:** Assume the current era is 2024-2025. If the year is ambiguous (e.g. "23"), prefer 2024 or 2025.

### 4. STRICT OUTPUT
- Output only the raw JSON object. No markdown, no explanations.
- If a field (like time) is not explicit, return `null`.

JSON STRUCTURE:
{
    "store_name": "string",
    "date": "YYYY-MM-DD",
    "time": "HH:MM",
    "total_amount": number,
    "subtotal": number,
    "tax_tps_amount": number,
    "tax_tvq_amount": number,
    "items": [
        { "desc": "string", "qty": number, "price": number }
    ]
}
"""


class OllamaVisionOCR:
    def __init__(self, host="http://173.209.56.38:11434"):
        self.client = Client(host=host)
        # Ensure this matches the tag you pulled on the server
        self.model = "qwen2.5vl:7b"
        print(f"👁️ Vision Engine: Connected to Ollama ({self.model})")

    def parse(self, image_bytes: bytes) -> dict:
        """
        Sends image bytes directly to the Vision Model (Qwen-VL).
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': VISION_PROMPT,
                    'images': [image_bytes]
                }]
            )

            content = response['message']['content']

            # --- Robust JSON Extraction ---
            # 1. Try finding JSON between code blocks
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # 2. Try finding raw JSON structure { ... }
            match = re.search(r'(\{.*\})', content, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # 3. Fail gracefully
            return {
                "error": "Vision Parsing Failed",
                "raw_response": content
            }

        except Exception as e:
            return {"error": f"System Error: {str(e)}"}