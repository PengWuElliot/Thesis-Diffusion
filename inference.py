from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import os
from diffusers.utils import load_image

# 初始化模型
model_path = "/scratch/work/wup5/rustoil/test/saved_model"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path)
pipe.to("cuda")  # 如果没有GPU可以改为 "cpu"

# 加载原图
init_image = Image.open("/scratch/work/wup5/dataset/inference/broke/001.png").convert("RGB")

# 设置 prompt 和 mask 路径
prompt = "surface break"
mask_dir = "/scratch/work/wup5/dataset/inference/broke/mask"

# 自动获取 mask 文件夹下的所有 PNG 文件，按文件名排序
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

# 输出文件夹
output_dir = "./output_inpainted"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个 mask，运行 inpainting
for idx, mask_file in enumerate(mask_files):
    mask_path = os.path.join(mask_dir, mask_file)
    mask_image = load_image(mask_path).convert("RGB")  # 保证 mask 是 RGB 图像

    output = pipe(prompt=prompt, image=init_image, mask_image=mask_image)

    # 保存结果
    output_path = os.path.join(output_dir, f"inpainted_{idx+1:03d}.png")
    output.images[0].save(output_path)
    print(f"Saved: {output_path}")

