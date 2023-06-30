import pandas as pd
import json
import torch
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,pipeline
from flask import Flask, request
import multiprocessing as mp
import concurrent.futures

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# @app.route('/qa', methods=['POST'])
# def predict():
# 加载预训练模型
model = AutoModelForQuestionAnswering.from_pretrained('./model/chinese_pretrain_mrc_roberta_wwm_ext_large')
tokenizer = AutoTokenizer.from_pretrained('./model/chinese_pretrain_mrc_roberta_wwm_ext_large')
qa_pipeline = pipeline("question-answering",model=model,tokenizer=tokenizer,device=device)

# sentence = request.form.get("key")
# 加载数据集
df = pd.read_json("datasets/contend.json")

# 转换数据集格式
dataset = []

# 创建一个生成器表达式

# dataset = ({'context': p['context'], 'question': q['question']} for p in df['data'] for q in p['paragraphs'] for q in p['qas'])

dataset = ({'context': q['context'], 'question': k['question']} for p in df['data'] for q in p['paragraphs'] for k in q['qas'])


# 使用pipeline方法进行预测
def qa(data):
    results = qa_pipeline(data)
    return results

# 使用进程池并行处理数据集中的每个问题并设置chunsize使用较小的批处理大小
# with mp.Pool() as pool:
#     results = pool.imap(qa, dataset,chunksize=10)

# 使用线程池并行处理数据集中的每个问题并设置chunsize使用较小的批处理大小
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = pool.map(qa, dataset,chunksize=10)

# 将结果保存到文件
output = []
for idx,result in enumerate(results):
    output.append({"idx":idx+1,"answer":result["answer"],"score":result["score"]})

with open("results.json",'w',encoding="utf-8") as f:
    json.dump(output,f,ensure_ascii=False,indent=4)
            
# if __name__ == '__main__':
#     app.run(
#         host='0.0.0.0',
#         port=6000
#     )
