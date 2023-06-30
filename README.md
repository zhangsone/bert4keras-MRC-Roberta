# 基于bert4keras的抽取式MRC基准代码
## 简介
本仓库是基于BERT4keras的抽取式MRC问答基础代码。详细介绍请看博客：https://kexue.fm/archives/8739

## 文件介绍
### datasets

该文件夹存放的是抽取式MRC数据集，分为训练集（train.json）、验证集（dev.json）以及测试集（test.json）。训练集和验证集是对模型在 下游任务中进行微调，使其可以学习到该领域的数据特征，模型在训练集和验证集上训练完成后，会生成一个模型的权重信息即xxx.weights。然后通过使用模型生成的权重信息，在测试集上进行相应的测试。

训练集的格式如下图所示：

![](README.assets/数据集示例图.png)

### model

该文件存放的是预训练模型，可根据自己需要选择相应的预训练模型，其中有bert、Roberta以及wwm等。

### src

该文件下存放的是源代码，其中`cmrc2018.py`是实现抽取式MRC的源代码，主要包含以下几部分：

- 加载数据集，生成每个batch_size的数据，主要由`load_data`函数和`data_generator`类进行实现

- 构建模型，该部分在代码有注释说明，一般情况下不改变模型的网络结构就不需要进行更改。
- 开始训练模型，并保存验证集中准确度最好的模型
- 测试模型效果

`snippets.py`是模型的配置文件，用来配置数据集的路径、模型的通用参数以及预训练模型的路径等。

`cmrc2018_evaluate.py`是用来测试模型生成答案的EM指标和F1指标。

`weights`文件夹用来保存模型生成的权重信息。

`results`文件夹用来保存模型测试生成的答案。

## 使用步骤

- 配置环境。需要的包已经列在`pip_requirements.txt`和`conda_requirements.txt`中。

  ```
  conda create -n bert4keras python=3.6
  source activate bert4keras
  ```

- 下载预训练模型。这里提供一个基础的[BERT模型下载链接](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)。将其下载放入model文件夹中。您也可以下载其他的预训练模型。

- 准备好数据集。请注意，如果您需要在下游任务中进行微调，请准备好train.json文件和dev.json文件。

  例如，您需要在反恐领域运行该代码，并且期望模型表现较好，您需要首先准备好反恐领域的文本，每一条文本数据的长度介于150字到900字之间，然后针对于每条文本，提出三到五个问题，并在文本中找出相应的答案，同时给出答案首次出现在文中的序号。所有的文本都标注完成之后，将其处理成相应的数据集格式即可。

  如果您只想让模型通过问题和文本，预测出相应的答案，您只需要准备好test.json文件夹，然后在通用领域抽取式阅读理解数据集（如：cmrc2018）上进行微雕，保存相应的权重信息。最后使用模型进行预测即可。

- 运行代码

  ```
  python cmrc2018.py
  ```

## 环境
- 软件：bert4keras>=0.10.8，具体请看`pip_requirements.txt`和`conda_requirements.txt`
- 硬件：显存不够，可以适当降低batch_size，如果有多GPU，可以开启多GPU进行训练
