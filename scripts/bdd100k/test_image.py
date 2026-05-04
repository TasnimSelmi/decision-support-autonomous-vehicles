from PIL import Image
from src.vlm.inference import call_vlm
from src.vlm.prompt import DRIVING_PROMPT

IMAGE_PATH = r"C:\Users\user\Desktop\VLM_RL project\data\samples\test_images\02ae4a52-e4715e65.jpg"


def main():
    image = Image.open(IMAGE_PATH).convert("RGB")

    print("Running Qwen2.5-VL on one image...\n")
    output = call_vlm(image, DRIVING_PROMPT)

    print("===== RAW OUTPUT =====")
    print(output)


if __name__ == "__main__":
    main()