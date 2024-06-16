import json
import torch
import numpy as np
from bert import Bert
from utils import convert_examples_to_features
from transformers import BertTokenizer
from run_ernie import ErnieConfig


model_name = 'ernie'
model_path = '../my_model/best_ernie.pkl'
config = ErnieConfig()
model= Bert(config).cuda()
model.load_state_dict(torch.load(model_path))
model.eval()
tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file)

with open('../data/KUAKE/KUAKE-QQR_test.json', 'r', encoding='UTF-8') as input_data, \
        open('../prediction_result/{}_pred.json'.format(model_name), 'w', encoding='UTF-8') as output_data:
    json_content = json.load(input_data)
    # 逐条读取记录，并将预测好的label赋值
    for block in json_content:
        query1 = block['query1']
        query2 = block['query2']
        feature = convert_examples_to_features(
            examples=[[block['query1'],block['query2'], 0]],
            tokenizer=tokenizer,
            max_length=config.pad_size,
            data_type='test'
        )
        feature=feature[0]
        input_ids = torch.tensor(np.array(feature.input_ids)).unsqueeze(0).cuda()
        attention_mask = torch.tensor(np.array(feature.attention_mask)).unsqueeze(0).cuda()
        token_type_ids = torch.tensor(np.array(feature.token_type_ids)).unsqueeze(0).cuda()
        output, loss = model(input_ids, attention_mask, token_type_ids,labels=None)
        output = torch.max(output.data, 1)[1].cpu().numpy()
        block['label'] = str(*output)
        # 写json文件
    json.dump(json_content, output_data, indent=2, ensure_ascii=False)