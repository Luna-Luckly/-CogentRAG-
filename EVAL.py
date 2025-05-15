import json
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score, recall_score, f1_score

model_name = ""  # 这里可以选择你想使用的预训练模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


import os

model_path_en2zh = ""
if not os.path.isdir(model_path_en2zh):
    print(f"错误: 目录 {model_path_en2zh} 不存在！")
else:
    print(f"目录 {model_path_en2zh} 存在，准备加载模型...")

model_en2zh = AutoModelForSeq2SeqLM.from_pretrained(model_path_en2zh)
tokenizer_en2zh = AutoTokenizer.from_pretrained(model_path_en2zh)

model_path_zh2en = ""
if not os.path.isdir(model_path_zh2en):
    print(f"错误: 目录 {model_path_zh2en} 不存在！")
else:
    print(f"目录 {model_path_zh2en} 存在，准备加载模型...")

model_zh2en = AutoModelForSeq2SeqLM.from_pretrained(model_path_zh2en)
tokenizer_zh2en = AutoTokenizer.from_pretrained(model_path_zh2en)



with open('.jsonl', 'r', encoding='utf-8') as f:
    content = f.read()
objects = content.splitlines()
standard_answers = [json.loads(line) for line in objects if line.strip()]


with open('.json', 'r', encoding='utf-8') as f:
    generated_answers = json.load(f)


def preprocess_answer(answer):
    return answer.strip().upper()


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # 使用模型的最后一层隐藏状态的平均值作为句子嵌入
    return embeddings.squeeze().cpu().numpy()


def translate_text(text, target_lang="en"):
    if target_lang == "en":
        inputs = tokenizer_zh2en(text, return_tensors="pt", padding=True, truncation=True).to(model_zh2en.device)
        translated = model_zh2en.generate(**inputs)
        translated_text = tokenizer_zh2en.decode(translated[0], skip_special_tokens=True)
        return translated_text
    else:
        inputs = tokenizer_en2zh(text, return_tensors="pt", padding=True, truncation=True).to(model_en2zh.device)
        translated = model_en2zh.generate(**inputs)
        translated_text = tokenizer_en2zh.decode(translated[0], skip_special_tokens=True)
        return translated_text


#y_true = [preprocess_answer(item["rationale"] + " " + item["correct"]) for item in standard_answers]
y_true = [preprocess_answer(item["output"]) for item in standard_answers]
y_pred = [preprocess_answer(item["answer"]) for item in generated_answers]


y_true_translated = [translate_text(ans, target_lang="en") for ans in y_true]
y_pred_translated = [translate_text(ans, target_lang="en") for ans in y_pred]


em_scores = [int(pred == true) for pred, true in zip(y_pred_translated, y_true_translated)]
mean_em = sum(em_scores) / len(em_scores)


cosine_similarities = []
correct_predictions = []
threshold = 0.50

for true_ans, pred_ans in zip(y_true_translated, y_pred_translated):
    emb_true = torch.tensor(get_embedding(true_ans))
    emb_pred = torch.tensor(get_embedding(pred_ans))
    
    similarity = F.cosine_similarity(emb_true.unsqueeze(0), emb_pred.unsqueeze(0)).item()  # 计算余弦相似度
    cosine_similarities.append(similarity)
    correct_predictions.append(similarity >= threshold)  # 认为达到阈值的为正确预测


accuracy = accuracy_score(correct_predictions, [True] * len(correct_predictions))
recall = recall_score(correct_predictions, [True] * len(correct_predictions), average='micro')
f1 = f1_score(correct_predictions, [True] * len(correct_predictions), average='micro')


def generate_counterfactuals(text):
    """生成对抗样本：同义替换+回译"""
    translated = translate_text(text, target_lang="zh")  # 先翻译成中文
    back_translated = translate_text(translated, target_lang="en")  # 再翻译回英文
    return preprocess_answer(back_translated)


cfr_similarities = []
for pred_ans in y_pred_translated:
    counterfactual_pred = generate_counterfactuals(pred_ans)
    
    emb_counterfactual = torch.tensor(get_embedding(counterfactual_pred)).unsqueeze(0)  # 添加 batch 维度
    emb_original = torch.tensor(get_embedding(pred_ans)).unsqueeze(0)  # 添加 batch 维度

    print(f"emb_counterfactual shape: {emb_counterfactual.shape}")
    print(f"emb_original shape: {emb_original.shape}")

    cfr_similarity = F.cosine_similarity(emb_counterfactual, emb_original).item()
    cfr_similarities.append(cfr_similarity)


cfr_score = sum(cfr_similarities) / len(cfr_similarities)



mean_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)


results = {
    "Accuracy": [accuracy],
    "Recall": [recall],
    "F1 Score": [f1],
    "Mean Cosine Similarity": [mean_cosine_similarity],
    "Exact Match (EM)": [mean_em],
    "Counterfactual Robustness (CFR)": [cfr_score]
}


