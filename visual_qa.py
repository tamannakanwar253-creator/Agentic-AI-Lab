from pathlib import Path
from PIL import Image
from langchain_ollama import ChatOllama

IMAGE_PATH = Path("images/sample.jpg")

print("=" * 65)
print("       IMAGE PROCESSING / VISUAL QA SYSTEM")
print("=" * 65)

# 1. Load image
print("\n[1] Loading image...")

if not IMAGE_PATH.exists():
    print("ERROR: Image not found:", IMAGE_PATH)
    raise SystemExit

image = Image.open(IMAGE_PATH)

print("Image found successfully.")

# 2. Extract image information
print("\n[2] Processing image...")

width, height = image.size
image_format = image.format
color_mode = image.mode

print("Width      :", width, "pixels")
print("Height     :", height, "pixels")
print("Format     :", image_format)
print("Color mode :", color_mode)

# 3. Load Llama
print("\n[3] Loading Llama 3.2...")

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("Llama 3.2 loaded.")

# 4. Question
question = input("\nEnter your question about the image: ")

# 5. Create context
context = f"""
Image file: {IMAGE_PATH.name}
Width: {width} pixels
Height: {height} pixels
Format: {image_format}
Color mode: {color_mode}
"""

# 6. Ask Llama
print("\n[4] Generating answer...")
print("-" * 65)

prompt = f"""
You are part of an Image Processing and Visual QA system.

Python extracted this information from the image:

{context}

User question:
{question}

Answer using only the information provided above.

The LLM cannot directly see the image.
Do not invent objects, people, text, or scenes.

If the information is insufficient, say:
"Insufficient image information."
"""

response = llm.invoke(prompt)

# 7. Display answer
print("\n[5] ANSWER")
print("=" * 65)
print(response.content)

print("\n" + "=" * 65)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 65)