from google import genai
from PIL import Image
import json
import os

api_key = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key)

p = r'C:\Users\MISHA JAISWAL\.gemini\antigravity\brain\24fb9615-7e0b-44eb-9e81-657840a0249e\.user_uploaded\media_1793541210316_9736fa30.jpg'
img = Image.open(p)

prompt = """
You are the AI Vision Core of EcoVision AI waste segregation system.
Analyze the provided waste image.
Classify the waste into exactly one of these 7 taxonomy classes:
[plastic, metal, cardboard, glass, paper, general_trash, organic].

Identify the specific physical items present, explain why it belongs to that class, and provide a normalized bounding box [ymin, xmin, ymax, xmax] (0 to 1000).

Return ONLY a valid JSON object matching this schema:
{
  "class_name": "organic",
  "confidence": 0.98,
  "detected_items": ["eggshells", "citrus peel", "lime peel"],
  "explanation": "Raw food scraps and eggshells are compostable wet organic waste.",
  "box_2d": [ymin, xmin, ymax, xmax]
}
"""

resp = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[img, prompt]
)

print('Gemini Vision output:')
print(resp.text)

