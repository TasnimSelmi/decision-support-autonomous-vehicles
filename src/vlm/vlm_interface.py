from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.vlm.config import DEVICE, MODEL_NAME, TORCH_DTYPE, MAX_NEW_TOKENS, DO_SAMPLE, TEMPERATURE
from src.vlm.prompt import VLM_PROMPT


DEFAULT_VLM_OUTPUT = {
    "hazard_level": "medium",
    "obstacle_presence": "yes",
    "collision_risk": "yes",
    "lane_safety": "unsafe",
    "crossing_pedestrian_presence": "no",
    "action_urgency": "slow",
}


def parse_vlm_response(raw_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return DEFAULT_VLM_OUTPUT.copy()


def normalize_vlm_output(raw_output: dict[str, Any] | None) -> dict[str, Any]:
    if not raw_output:
        return DEFAULT_VLM_OUTPUT.copy()

    normalized = DEFAULT_VLM_OUTPUT.copy()
    normalized.update({key: raw_output.get(key, normalized[key]) for key in normalized})
    return normalized


class QwenVLMClient:
    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._process_vision_info = process_vision_info
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=TORCH_DTYPE,
        )
        self._model.to(DEVICE)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(MODEL_NAME)

    def infer_image(self, image_path: str | Path) -> str:
        self._ensure_loaded()

        image_path = Path(image_path)
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

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = self._process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = {key: value.to(DEVICE) if hasattr(value, "to") else value for key, value in inputs.items()}

        import torch

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()


def run_vlm_inference(image_path: str | Path, client: QwenVLMClient | None = None) -> dict[str, Any]:
    if client is None:
        client = QwenVLMClient()

    try:
        raw_text = client.infer_image(image_path)
        return normalize_vlm_output(parse_vlm_response(raw_text))
    except Exception:
        return DEFAULT_VLM_OUTPUT.copy()
