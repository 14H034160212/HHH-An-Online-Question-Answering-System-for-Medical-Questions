# HBDA-An-Online-Question-Answering-System-for-Medical-Questions

## Knowledge graph establishment refer the following link
https://github.com/liuhuanyong/QASystemOnMedicalKG

## BERT
https://github.com/google-research/bert

## BiLSTM+Attention
https://zhuanlan.zhihu.com/p/31638132

https://github.com/likejazz/Siamese-LSTM

https://github.com/LuJunru/Sentences_Pair_Similarity_Calculation_Siamese_LSTM

## AttentionLayer is referred from the following link
https://github.com/uhauha2929/examples/blob/master/Hierarchical%20Attention%20Networks%20.ipynb

## Dataset
### Medical knowledge dataset collected from the following medical website
https://www.medicinenet.com/medterms-medical-dictionary/article.htm

https://www.nhsinform.scot/illnesses-and-conditions/a-to-z

### Model train_dev_test dataset are filter out from Quora question pair dataset.
https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs

### Medical question and answer pair dataset is referred from the following link.
https://github.com/LasseRegin/medical-question-answer-data



|Model|Average Eval_accuracy by three times|Range of change|
|:---|:---|:---|
|BERT baseline model|0.7686|(-0.0073, +0.0057)|
|HDBA model|**0.8146**|(-0.0082, +0.0098)|
|Bi-LSTM + Attention model|0.8043|(-0.0103, +0.0062)|

### The scale of knowledge graph about 700 diseases. For each disease, there exists symptom, accompany_disease, prevent_way, cure_way and totally 6 entities.
<img src="./Medical_knowledge_graph_establishment/System_screenshot/Figure_3.png" width="800" />

### System architecture
<img src="./Medical_knowledge_graph_establishment/System_screenshot/system_architecture.png" width="800" />
