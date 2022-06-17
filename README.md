# Capston Design 1 - Distilbert fine-tuning
Advertisement Classification by Distilbert and Tensorflow
by team 네츄럴, 20171645 parkchanwoo

## description
- fine-tuning "distilbert-base-uncased" by twitter dataset
- 1500+ tweets including top 10 coins keywords
- BATCH_SIZE = 2
- EPOCH_NUM = 10
- save fine-tuned model to "./adv_model"

# To run
distilbert.py and twitter.csv file must exist same path

-pip install pandas
-pip install tensorflow
-pip install tensorflow.keras
-pip install transformers
-pip install pickle

-python distilbert.py
