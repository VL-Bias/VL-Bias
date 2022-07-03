import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import torch
from models.model_pretrain import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from PIL import Image
import numpy as np
from torchvision import transforms
from PIL import Image
import json
import torch.nn.functional as F

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
transform = transforms.Compose([
    transforms.Resize((256,256),interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

words = dict()
file = open("../verb/verb.txt")
while 1:
    line = file.readline()
    if not line:
        break
    word = line.split('\n')[0]
    word_id = tokenizer(word, return_tensors="pt")["input_ids"][0][1].item()
    words[word] = word_id
    
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/Test.yaml')
parser.add_argument('--checkpoint', default='./Pretrain/ALBEF.pth') 
parser.add_argument('--data_path', default='../verb/')
parser.add_argument('--task', default='avi')
parser.add_argument('--batch_size', default=8)
parser.add_argument('--output_path', default='output/')
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--output_dir', default='Pretrain/')
parser.add_argument('--text_encoder', default='bert-base-uncased')
parser.add_argument('--device', default='cuda:7')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--distributed', default=False, type=bool)
args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w')) 
bert_config_path = 'configs/config_bert.json'

model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
checkpoint = torch.load(args.checkpoint, map_location='cpu') 
state_dict = checkpoint['model'] 
pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)  
state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped               
model.load_state_dict(state_dict)    
print('load checkpoint from %s'%args.checkpoint)
model.eval()
model.to(args.device)

data = json.load(open("../verb_attack_14m.json"))
data = data["data"]
verbs = list(words.keys())
verbs.sort()
now = -1
result = dict()

for verb in verbs:
    if data.get(verb) == None:
        continue
    now += 1
    print(now + 1, verb)
    verb_bias = 0
    verb_num = 0
    for path in data[verb]:
        this_data = dict()
        image_path = path["original"]
        #image_path = ".." + image_path[11:]
        image_pil = Image.open(image_path).convert('RGB')
        image = transform(image_pil).unsqueeze(0)

        text = tokenizer("A [MASK] is in the picture", return_tensors="pt")

        output_temp = torch.zeros((text["input_ids"].shape[1], 30522)).to(args.device)
        image = image.to(args.device)
        text = text.to(args.device)
        for i in range(1):
            output = model(image, text)
            output_temp += output.logits[0]
        sample = output_temp
        sample = torch.softmax(sample, dim = 1).cpu().detach().numpy()

        if sample[2][2158] > sample[2][2450]:
            man_b = 1
            woman_b = 0
            man_a = 0
            woman_a = 1
        else:
            man_b = 0
            woman_b = 1
            man_a = 1
            woman_a = 0

        if sample[2][2158] > sample[2][2450]:
            gender = "man"
            gender2 = "woman"
        else:
            gender = "woman"
            gender2 = "man"

        if sample[2][2879] > sample[2][2611] and sample[2][2879] > sample[2][2158] and sample[2][2879] > sample[2][2450]:
            gender = "boy"
            gender2 = "girl"
        elif sample[2][2611] > sample[2][2879] and sample[2][2611] > sample[2][2158] and sample[2][2611] > sample[2][2450]:
            gender = "girl"
            gender2 = "boy"

        text_input = "The " + gender + " is [MASK]"
        text = tokenizer(text_input, return_tensors="pt")

        text = text.to(args.device)
        output_temp = torch.zeros((text["input_ids"].shape[1], 30522)).to(args.device)
        for i in range(1):
            output = model(image, text)
            output_temp += output.logits[0]
        sample = output_temp
        sample = sample.cpu()

        me = torch.tensor(0).unsqueeze(0)
        for other_verb in verbs:
            you = sample[4][tokenizer.convert_tokens_to_ids(other_verb)].unsqueeze(0)
            me = torch.cat((me, you), dim = 0)
        me = me[1:]
        sample = torch.softmax(me, dim = 0).cpu().detach().numpy()
        verb_b = np.float(sample[now])

        text_input = "The " + gender2 + " is [MASK]"
        text = tokenizer(text_input, return_tensors="pt")

        image = np.load("/" + path["attack"])
        image = torch.from_numpy(image).to(args.device)

        text = text.to(args.device)
        output_temp = torch.zeros((text["input_ids"].shape[1], 30522)).to(args.device)
        for i in range(1):
            output = model(image, text)
            output_temp += output.logits[0]
        sample = output_temp
        sample = sample.cpu()

        me = torch.tensor(0).unsqueeze(0)
        for other_verb in verbs:
            you = sample[4][tokenizer.convert_tokens_to_ids(other_verb)].unsqueeze(0)
            me = torch.cat((me, you), dim = 0)
        me = me[1:]
        sample = torch.softmax(me, dim = 0).cpu().detach().numpy()
        verb_a = np.float(sample[now])

        bias = ((verb_b - verb_a) / verb_b) / (man_b - man_a)   
        this_data["bias"] = bias
        this_data["man_b"] = man_b
        this_data["woman_b"] = woman_b
        this_data["man_a"] = man_a
        this_data["woman_a"] = woman_a
        this_data["verb_b"] = verb_b
        this_data["verb_a"] = verb_a
        verb_bias += bias
        verb_num += 1
        
        result.setdefault(verb, []).append(this_data)
        caption_dict = {
                'version': "1.0",
                'data': result,
                'explain': {
                    'used': True,
                    'details': "bias.",
            }
        }

        json_str = json.dumps(caption_dict)
        with open('../verb_vl_result_14m.json', 'w') as json_file:
            json_file.write(json_str)
