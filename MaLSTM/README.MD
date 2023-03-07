This is the folder of MaLSTM referred from [[Code](https://github.com/likejazz/Siamese-LSTM)] [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/10350)].

## Installation

Download `corpora/stopwords` using the following way.

```
>>> import nltk
>>> nltk.download('stopwords')
```

## Training and evaluation

`python train.py`

## Testing on some examples

`python predict.py`

## Tensorboard Sentence Embedding Visualization (Only implemented in branch [`origin/tensorflow2`](https://github.com/14H034160212/HHH-An-Online-Question-Answering-System-for-Medical-Questions/tree/tensorflow2))

`tensorboard --logdir ./malstm/`

![image](https://user-images.githubusercontent.com/23516191/223430710-36733d92-ec62-4c62-97b5-3bfd5d80667f.png)
