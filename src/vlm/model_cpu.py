from __future__ import annotations

from pathlib import Path
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from config import MODEL_NAME, TORCH_DTYPE, DEVICE, MAX_NEW_TOKENS, DO_SAMPLE, TEMPERATURE
from prompt import VLM_PROMPT


class QwenVLInference:
    def __init__(self) -> None:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
        )
        self.model.to(DEVICE)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

    def infer_image(self, image_path: Path) -> str:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(DEVICE) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()