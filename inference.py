
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import os
from diffusers.utils import load_image

# 初始化模型
model_path = "/scratch/work/wup5/rustoil/rust/saved_model"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path)
pipe.to("cuda")  # 如果没有GPU可以改为 "cpu"

# 加载原图
init_image = Image.open("/scratch/work/wup5/dataset/inference/rust/000.png").convert("RGB")

# 设置 prompt 和 mask 路径
prompt = "rust"
mask_dir = "/scratch/work/wup5/dataset/inference/rust/mask"
mask_files = ["000.png", "001.png", "002.png", "003.png", "004.png", "005.png"]  # 你的三张 mask 文件名

# 输出文件夹
output_dir = "./output_inpainted"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个 mask，运行 inpainting
for idx, mask_file in enumerate(mask_files):
    mask_path = os.path.join(mask_dir, mask_file)
    mask_image = load_image(mask_path).convert("RGB")  # 保证 mask 是 RGB 图像

    output = pipe(prompt=prompt, image=init_image, mask_image=mask_image)

    # 保存结果
    output_path = os.path.join(output_dir, f"inpainted_{idx+1}.png")
    output.images[0].save(output_path)
    print(f"Saved: {output_path}")

