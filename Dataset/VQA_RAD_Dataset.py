import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class VQA_RAD_Dataset(Dataset):
    def __init__(self, json_path, image_folder_path, vision_processor, use_half_tensor = True, use_few_shot = False, use_adv_image = False):

        self.image_folder_path = image_folder_path
        self.vision_processor = vision_processor

        self.use_adv_image = use_adv_image

        self.use_half_tensor = use_half_tensor
        self.use_few_shot = use_few_shot

        # TODO: implement prompt wrapper class
        self.sys_prompt = "You are a helpful medical assistant. " \
                          "You are being provided with an image and a question about the image. " \
                          "Answer the question. "
        
        # the same few-shot images as demonstrated in the Med-Flamingo demo
        # all the questions are sampled from train split
        self.few_shot_image_folder = "../Data/VQA-RAD/few_shot_image/"
        self.few_shot_image_list = [
                                        'synpic50962.jpg',
                                        'synpic52767.jpg',
                                        'synpic30324.jpg',
                                        'synpic21044.jpg',
                                        'synpic54802.jpg',
                                        'synpic57813.jpg'
                                    ]
        
        data = pd.read_json(json_path)
        
        # use test data only
        data = data[data["phrase_type"].isin(["test_freeform", "test_para"])]
        data["answer"] = data["answer"].str.lower()

        self.data = data

        self.few_shot_prompt = None
        self.few_shot_image = None
        if self.use_few_shot:
            self._few_shot_init()
        
    # the same few-shot as demonstrated in the Med-Flamingo demo
    def _few_shot_init(self):
        self.sys_prompt = "You are a helpful medical assistant. " \
                          "You are being provided with images, a question about the image and an answer. " \
                          "Follow the examples and answer the last question."
        self.few_shot_prompt = "<image>Question: What is/are the structure near/in the middle of the brain? Answer: pons.<|endofchunk|>" \
                               "<image>Question: Is there evidence of a right apical pneumothorax on this chest x-ray? Answer: yes.<|endofchunk|>" \
                               "<image>Question: Is/Are there air in the patient's peritoneal cavity? Answer: no.<|endofchunk|>" \
                               "<image>Question: Does the heart appear enlarged? Answer: yes.<|endofchunk|>" \
                               "<image>Question: What side are the infarcts located? Answer: bilateral.<|endofchunk|>" \
                               "<image>Question: Which image modality is this? Answer: mr flair.<|endofchunk|>"
        few_shot_image_path = [os.path.join(self.few_shot_image_folder, p) for p in self.few_shot_image_list]
        few_shot_images_raw = [Image.open(path) for path in few_shot_image_path]
        few_shot_images = [self.vision_processor(img).unsqueeze(0) for img in few_shot_images_raw]
        self.few_shot_images = torch.cat(few_shot_images, dim=0)

        if self.use_half_tensor:
            self.few_shot_images = self.few_shot_images.type(torch.HalfTensor)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        question  = sample['question']
        anwser = sample['answer']
        answer_type = sample['answer_type']
        qid = str(sample['qid']) # to distinguish adversarial image
        
        image_path = self.image_folder_path + sample['image_name']
        if self.use_adv_image:
            image_path = self.image_folder_path + qid + "_adv_" + sample['image_name']

        image = self.vision_processor(Image.open(image_path)).unsqueeze(0)

        if self.use_half_tensor:
            image = image.type(torch.HalfTensor)

        prompt = self.sys_prompt
        if self.use_few_shot:
            image = torch.cat((self.few_shot_images, image), dim=0)
            prompt += self.few_shot_prompt
        
        item = {
            'qid': qid,
            'prompt': prompt + "<image>Question: " + question + " Answer:",
            'question': question,
            'image_name': sample['image_name'],       
            'image': image,
            'label': anwser,
            'answer_type': answer_type,
        }
        return item
        
