#-*- coding:utf-8 -*-
import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from flask import Flask, request
import multiprocessing as mp
import concurrent.futures
import logging

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

# 创建一个日志处理器，输出到控制台
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 将日志处理器添加到应用程序日志处理器列表中
app.logger.addHandler(handler)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# @app.route('/qa', methods=['POST'])
# def predict():
# 加载预训练模型
model = AutoModelForQuestionAnswering.from_pretrained('./model/chinese_pretrain_mrc_roberta_wwm_ext_large')
tokenizer = AutoTokenizer.from_pretrained('./model/chinese_pretrain_mrc_roberta_wwm_ext_large')
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

# sentence = request.form.get("key")
# 加载数据集



# 使用pipeline方法进行预测
def qa(data):
    results = qa_pipeline(data)
    return results


# 使用进程池并行处理数据集中的每个问题并设置chunsize使用较小的批处理大小
# with mp.Pool() as pool:
#     results = pool.imap(qa, dataset,chunksize=10)

# 使用线程池并行处理数据集中的每个问题并设置chunsize使用较小的批处理大小
@app.route("/qarob",methods=["POST"])
def run():
    file_path = request.form.get("in_file")
    df = pd.read_json(file_path)
    # 转换数据集格式
    dataset = ({'context': q['context'], 'question': k['question']} for p in df['data'] for q in p['paragraphs'] for k
               in
               q['qas'])
    # 创建一个包含所有id的列表
    ids = (k['id'] for p in df['data'] for q in p['paragraphs'] for k in q['qas'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(qa, dataset, chunksize=10)
    return_result = {}
    for idx, result in zip(ids, results):
        if result["answer"] is not None:
            if result["score"] > 0.00:
                return_result[idx] = result["answer"]
            else:
                return_result[idx] = "未找到相关信息"
            app.logger.info(result["score"])
            app.logger.info(idx)
    app.logger.info(return_result)
    return json.dumps(return_result,ensure_ascii=False)
    


# 将结果保存到文件

def result_(results):
    output = []
    for idx, result in enumerate(results):
        output.append({"idx": idx + 1, "answer": result["answer"], "score": result["score"]})

    with open("results.json", 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=6005
    )

