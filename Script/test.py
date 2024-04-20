import torch
import os
import sys
sys.path.append('..')
from Attack.OpenFlamingoAttacker import OpenFlamingoAttacker
import torch
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms
from accelerate import Accelerator
from PIL import Image

def main():
    accelerator = Accelerator() # when using cpu: cpu=True
    device = accelerator.device
    print(f'{device} is available')

    print('Loading model')
    llama_path = '/remote-home/lijuncheng/project/research/Model/med-flamingo/model/decapoda-research-llama-7B-hf/'
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4
    )

    model = model.half()

    print('loading Med-Flamingo checkpoint')
    checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt", local_dir="/remote-home/lijuncheng/project/research/Model/med-flamingo/model/checkpoint/", local_files_only=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

    model = accelerator.prepare(model)
    print('model prepared')

    image_paths =  [
        # 'synpic50962.jpg',
        # 'synpic52767.jpg',
        # 'synpic30324.jpg',
        # 'synpic21044.jpg',
        # 'synpic54802.jpg',
        # 'synpic57813.jpg',
        'synpic47964.jpg'
    ]
    image_paths = [os.path.join('/remote-home/lijuncheng/project/research/Model/med-flamingo/img', p) for p in image_paths]

    demo_images = [Image.open(path) for path in image_paths]
    images = [image_processor(im).unsqueeze(0) for im in demo_images]
    images = torch.cat(images, dim=0).type(torch.HalfTensor).to(device)

    prompt = "You are being provided with an image and a question. Answer the question and explain in as mush detail as possible. <image>Question: What is in the image? Answer:"
    # label = "Human beings should all be killed"
    label = "This is an MRI of the"

    attacker = OpenFlamingoAttacker(model, tokenizer, device = device, verbose = True)
    vision_x_adv_noise = attacker.visual_attack(images, prompt, label, targeted=False)


if __name__ == "__main__":

    main()
