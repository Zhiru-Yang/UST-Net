

[toc]



# 开放域问答模型分类

## 1. Two-stage retriever-reader approaches

![image-20221120160006918](开放域问答.assets/image-20221120160006918.png)

### （1）基本框架

- **retriever**：从文档库中寻找可能包含答案的文档，通常是传统的稀疏向量空间方法，如 TF-IDF 或者 BM25

  - 通常不可训练
  - 在文档水平上搜寻，而非段落水平

- **reader**：在给定的文档或段落中找到答案，通常是神经阅读理解模型

  - 输入是文章P和问题Q，P也可以是段落
  - 输出是答案A，如果A是P的文本片段，则此任务为“抽取式问答”
  - 如果有多个P，则可能产生多个A，此时根据span score进行挑选

  ![image-20221120160339708](开放域问答.assets/image-20221120160339708.png)

### （2）Reader模型

-  [Reading Wikipedia to Answer Open-Domain Questions.pdf](开放域问答.assets\Reading Wikipedia to Answer Open-Domain Questions.pdf) （2017）

![image-20221120160706744](开放域问答.assets/image-20221120160706744.png)

-   [BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf](开放域问答.assets\BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf) （2018）

![image-20221120160724624](开放域问答.assets/image-20221120160724624.png)

### （3）改进

- **BERTserini**：引入段落级的Anserini Retriever，同时用segment和答案片段的加权分数挑选答案

![image-20221120164808724](开放域问答.assets/image-20221120164808724.png)

- **Multi-passage training**：预测时，不同段落抽取的答案难以比较，因此可以让来自不同段落的答案在正则阶段进行统一评分。
  - 每个段落在寻找答案span时是独立的，相互之间不可见
  - 只有在使用softmax进行评分时，才将各个段落的span联合计算

>  [Simple and Effective Multi-Paragraph Reading Comprehension.pdf](开放域问答.assets\Simple and Effective Multi-Paragraph Reading Comprehension.pdf) 
>
>  [Multi-passage BERT- A Globally Normalized BERT Model for Open-domain Question Answering.pdf](开放域问答.assets\Multi-passage BERT- A Globally Normalized BERT Model for Open-domain Question Answering.pdf) 

- **Training a passage re-ranker**：对retriever搜寻到的文章进行重排序，可使用强化学习方法
- **Training an answer re-ranker**：对answer进行重排序
  - **Strength**-based re-ranker：如果候选答案由多个高置信度的证据支持，则其更可能是正确的
  - **Coverage**-based re-ranker：如果候选答案的证据合集包含了问题中的大部分信息，则其更可能是正确的

![image-20221120170754627](开放域问答.assets/image-20221120170754627.png)

- **Hard EM Learning**：开放域QA的训练集常使用远程监督来获得，但我们并不知道哪个span是正确答案，因此可以使用这一方法解决

![image-20221120171416626](开放域问答.assets/image-20221120171416626.png)



## 2. Dense retriever and end-to-end training

![image-20221120173856791](开放域问答.assets/image-20221120173856791.png)

### （1）Why dense retrieval now?

1. dense很困难：需要对百万量级的文档、千万量级的段落进行编码、索引、搜索

2. 有标注数据不够：现在可以使用预训练模型BERT

3. MIPS(Maximum Inner Product Search, 最大内积搜索)：快速最大内积搜索技术得到了发展
  最近邻问题: 在X中找到一个p，使得p和q的点积在集合X中是最大的![image1.png](开放域问答.assets/image1.png)

  对于MIPS来说，则是在X中找到一个p，使得p和q的距离最小 ![屏幕截图 2023-02-20 091453.png](D:\微信接收\WeChat Files\wxid_otr1qtjxkk8y22\FileStorage\File\2023-02\开放域问答(1)\开放域问答.assets\屏幕截图 2023-02-20 091453.png)

### （2）ORQA

>  [Latent Retrieval for Weakly Supervised Open Domain Question Answering.pdf](开放域问答.assets\Latent Retrieval for Weakly Supervised Open Domain Question Answering.pdf) （2019）

1. **ORQA**：即Open-Retriever Question Answering，第一个联合学习retriever和reader的模型。

![image-20221120180805898](开放域问答.assets/image-20221120180805898.png)



2. **ICT**：即Inverse Cloze Task，用于retrieval的预训练。给定一个句子，让模型预测其上下文。

> 以下图为例，模型从文本片段中随机抽取一个句子作为伪查询，其上下文作为伪证据文本：“...Zebras have four gaits: walk, trot, canter and gallop. **They are generally slower than horses, but their great stamina helps them outrun predators.** When chased, a zebra will zigzag from side to side...”

![image-20221120181058208](开放域问答.assets/image-20221120181058208.png)

> 仍然鼓励retriever学习词语匹配，最终模型仅在90%的样本中删除原句

![image-20221120182231308](开放域问答.assets/image-20221120182231308.png)

> 此外，BERT~B~在预训练结束后将被固定，以便预计算block representation，并提升搜索效率

### （3）REALM

>  [Retrieval Augmented Language Model Pre-Training.pdf](开放域问答.assets\Retrieval Augmented Language Model Pre-Training.pdf) （2020）

1. **REALM**：即Retrieval-augmented Language Model，同时在retriever和reader上进行预训练

![image-20221120183642334](开放域问答.assets/image-20221120183642334.png)

2. **MLM**：掩码语言模型

   1. 冷启动问题：使用ICT作为第一阶段的预训练
   2. Salient span masking：仅掩码命名实体与日期

   > NQ上的评分：
   >
   > - Random uniform masks: 32.3
   > - Random span masks: 35.3
   > - Salient span masks: 38.2

   3. 可以使用比Wikipedia更大的语料进行预训练

   > CC-News vs Wikipedia：40.4 vs 39.2

   4. 可以异步更新证据编码器

<img src="开放域问答.assets/image-20221120183744431.png" alt="image-20221120183744431" style="zoom:33%;" />

### （4）DPR

>  [Dense Passage Retrieval for Open-Domain Question Answering.pdf](开放域问答.assets\Dense Passage Retrieval for Open-Domain Question Answering.pdf) 

1. **DPR**: Dense Passage Retrieval，从少量的问答对中训练出一个稠密的retriever，**无需预训练**

![image-20221120185100132](开放域问答.assets/image-20221120185100132.png)

2. **正样本**

   1. 在阅读理解数据集中提供的正样本
   2. 包含答案字符串的BM25得分最高的文章

3. **负样本**

   1. Random：语料中的随机文章
   2. BM25：不包含答案字符串的BM25得分最高的文章
   3. Gold：其他问题的正样本（使用批次内负样本可以提高训练效率）

   <img src="开放域问答.assets/image-20221120190303369.png" alt="image-20221120190303369" style="zoom:33%;" />

4. **表现**
   1. DPR在NaturalQuestions、WebQuestions、TREC和TriviaQA上好于BM25，但在SQuAD表现略差
   2. DPR在大数据集上（NQ）比REALM好，但在小数据集上WebQ，TREC上需要联合大数据集进行训练才能超过REALM

5. **重要结论**
   1. 至少对于中等大小的QA数据集，不需要预训练
   2. retriever和reader联合训练的效果并不好于pipline训练，且pipeline只需要索引一次，因而效率更高
   3. ~~阅读理解与问答数据集相比区别不大，可以仅使用问答对训练系统~~

### （5）RAG

>  [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf](开放域问答.assets\Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf) 

1. **RAG**：即Retrieval-Augmented Generation

![image-20221120193659965](开放域问答.assets/image-20221120193659965.png)

2. **基本结构**
   1. Retrieval model： $p(z|x)$，本文使用DPR
   2. seq2seq model：$p(y|x,z)$，本文使用BART

### （6）模型比较

![image-20221120194336670](开放域问答.assets/image-20221120194336670.png)

## 3. Retriever-free approaches

### （1）隐式的retriever

- 我们能不能使用PLM（pre-trained language model）作为知识库？
- 我们能不能直接从PLM中获得答案？
- PLM是在大规模语料上训练的，那么他们应该能记忆相当的知识。

### （2）GPT-2

- GPT-2是一个基于transformer的生成式语言模型，在大规模语料上进行训练

![image-20221120192121931](开放域问答.assets/image-20221120192121931.png)

- GPT-2 Zero-Shot在部分问题上有着极高的正确率

![image-20221120192159429](开放域问答.assets/image-20221120192159429.png)

- 但GPT-2的Zero-Shot相比有监督系统仍然十分差

![image-20221120192457393](开放域问答.assets/image-20221120192457393.png)

### （3）GPT-3 (2020.5)

- GPT-3暴力出奇迹，参数高达1750亿，不可能再进行微调。但是GPT-3的Few-Shot效果极好。

![image-20221120192707808](开放域问答.assets/image-20221120192707808.png)

### （4）T5

![image-20221120192852029](开放域问答.assets/image-20221120192852029.png)

