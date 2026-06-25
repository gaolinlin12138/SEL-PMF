import os
import json
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

device = "cuda:0"
model_name = "Hugging-Face/Qwen3-VL-4B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)

root_dir = "weibo_dataset"
subdirs = ["nonrumor_images", "rumor_images"]

prompt = """
作为一名专注于视觉理解和虚假信息检测的多媒体内容分析专家，你将收到一张新闻图像，需要对其进行准确、客观的描述。请提供一段简洁的图像说明，重点关注图中关键视觉要素，例如人物、物体、背景和任何可见文字。务必逐字转录图像中出现的文字并原样呈现。确保描述简明、清晰且避免任何主观判断。保持中立且分析性的语气，类似专业媒体分析师的风格。
"""

# prompt = """
# As an expert in multimedia content analysis specializing in visual understanding and fake news detection, you will receive a news image that requires an accurate and objective description. Please provide a concise caption focusing on the key visual elements in the image, such as people, objects, background, and any visible text. Be sure to transcribe all visible text in the image exactly as it appears. Keep the description clear, factual, and free of subjective judgments. Maintain a neutral and analytical tone, similar to that of a professional media analyst.
# """


output_file = "weibo_captions.json"
results = []


def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def generate_caption(image_path):
    try:
        image = preprocess_image(image_path)
        if image is None:
            return "Error: Failed to load image"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None
            )

        generated_ids = output[0][inputs.input_ids.shape[1]:]
        caption = processor.decode(generated_ids, skip_special_tokens=True).strip()

        return caption

    except Exception as e:
        return f"Error processing image: {e}"


import torch

for folder in subdirs:
    folder_path = os.path.join(root_dir, folder)
    label = 0 if folder == "nonrumor_images" else 1

    for img_name in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
        img_path = os.path.join(folder_path, img_name)
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            continue

        caption = generate_caption(img_path)

        results.append({
            "image": img_path,
            "label": label,
            "caption": caption
        })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 图像描述已完成，保存于 {output_file}")
