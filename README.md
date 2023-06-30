# 基于bert4keras的抽取式MRC基准代码
## 简介
本仓库是基于roberta的抽取式问答基础代码。

## 文件介绍
### datasets

该文件夹存放的是抽取式数据集，分为训练集（train.json）、验证集（dev.json）以及测试集（test.json）。训练集和验证集是对模型在 下游任务中进行微调，使其可以学习到该领域的数据特征，模型在训练集和验证集上训练完成后，会生成一个模型的权重信息即xxx.weights。然后通过使用模型生成的权重信息，在测试集上进行相应的测试。

训练集的格式如下图所示：

![](README.assets/数据集示例图.png)

### model

该文件存放的是预训练模型，可根据自己需要选择相应的预训练模型。


## 使用步骤

- 配置环境。需要的包已经列在`requirements.txt`中。

- 下载预训练模型。这里提供一个基础的([https://huggingface.co/luhua/chinese_pretrain_mrc_roberta_wwm_ext_large])。将其下载放入model文件夹中。您也可以下载其他的预训练模型。

- 准备好数据集。请注意，如果您需要在下游任务中进行微调，请准备好train.json文件和dev.json文件。

  如果您只想让模型通过问题和文本，预测出相应的答案，您只需要准备好test.json文件夹，然后在通用领域抽取式阅读理解数据集（如：cmrc2018）上进行微雕，保存相应的权重信息。最后使用模型进行预测即可。

- 运行代码

  ```
  python predict_qa_roberta.py
  ```

