# %% [markdown]
# ## Online banking fraud detection using natural language processing techniques

# %% [markdown]
# **Team 9:**
# 
# - Latif Masud
# - ​Wesley Mitchell
# - Gerald Wagner​
# 
# **Course:** AI 574 – Natural Language Processing (Spring 2025)

# %% [markdown]
# ### Problem Statement
# * This project aims to identify fraudulent activity in online banking transactions using Natural Language Processing techniques. Online banking activity can be monitored by the webpages or API endpoints a user interacts with throughout their entire session history. With this sequence of user actions, a binary classification can be trained such that it labels the activity as valid or fraudulent; if fraudulent, remediation steps could then be implemented such as denying the transaction. With online banking a staple of people every day financial lives and 100's of millions of dollars transacted daily, identifying fraudulent activity is of upmost importance to prevent unnecessary monetary losses for both individuals and financial institutions.
#     
# * **Keywords:** Online banking, fraud, fraud detection, financial industry 

# %% [markdown]
# ### Data Collection
# 
# * Source(url): https://github.com/pboulieris/FraudNLP/blob/master/Fraud%20Detection%20with%20Natural%20Language%20Processing.rar
# * Short Description: The data set of 105,303 online banking transactions with 9 transaction characteristics:
#     * Action time mean: the average time between actions in a transaction
#     * Action time std: the standard deviation of the time between actions
#     * log(amount): the natural logarithm of the transaction amount
#     * Transaction Type: a string indicating whether the transaction is fraudulent or not
#     * time_to_first_action: the time between the start of the transaction and the first action taken
#     * actions_str: a string containing the names of all actions taken in the transaction
#     * total_time_to_transaction: the total time elapsed from the start of the transaction to its completion
# 
# * Keywords: bank transactions, user actions, API endpoints, webpage urls, dollar amount

# %% [markdown]
# ### Required packages
# 
# * the following packages are required to run this notebook:
#     * pandas
#     * scikit-learn
#     * nltk
#     * matplotlib
# 
# Install by creating and activating a virtual environment, then installing via the pip command:
# 
# !pip install pandas scikit-learn nltk matplotlib wordcloud

# %% [markdown]
# ### Imports

# %%
# !pip install -r requirements.txt

# %%
from pathlib import Path

import matplotlib.pyplot as plt
from nltk import FreqDist
import pandas as pd
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# %% [markdown]
# ### Load the data

# %%
# online banking transaction data
path = Path('./data/Fraud Detection with Natural Language Processing.pkl')
df = pd.read_pickle(path)

# %%
# vocabulary of API calls
path_vocab = Path('./data/vocab.csv')
df_vocab = pd.read_csv(path_vocab)

# %% [markdown]
# ### Exploratory data analysis (EDA)

# %%
df.shape

# %%
df.head(10)

# %%
df.describe()

# %%
df.dtypes

# %%
df.groupby('is_fraud').describe()

# %%
df_vocab.head()

# %%
df_vocab.shape

# %%
df_vocab.describe()

# %% [markdown]
# EDA Summary
# 
# - the transaction dataset contains 105303 online banking transactions
# - of the 105303 transactions, 105202 are valid while only 101 are fraudulent
#     - this is a severe class imbalance that will have to be handled in the neural network architecture
# - there are 9 attributes for each banking transaction:
#     - a label for trasactions that are valid or fraudulent (0 or 1 respectively)
#     - list of user actions encoded as a list of integers which corresponds to the vocabulary dataframe
#     - list of times in ms for each user action to occur
#     - the total elapsed time of the transaction in ms
#     - Recency, Frequency, and Monetary features:
#         - the transaction amount in log(Euros)
#         - the device characteristics
#         - the IP address of the user
#         - the beneficiary's frequency of conducting a transaction
#         - the applications used for the transaction (i.e., Android or iOS)
# - there also exists a vocabulary dataset which contains a list of API endpoints/webpage urls which a user can access
#     - these are used to translate the encoded user action column of the transaction dataset back to the original url's
#     - there are 1916 total endpoints/url's, all of which are unique
#     - the index of the dataframe corresponds to the id value used in the user action list from the transaction dataframe

# %% [markdown]
# ### Data Preprocessing
# 
# * Enumerate and present the main steps you preformed in the data preprocessing
# * Add your code and interpret the outcome of main steps/functions

# %%
# dictionary mapping ids in transaction data to vocabulary
vocab = df_vocab['Name'].to_list()

vocab_sentences = []
for endpoint in vocab:
    sentence = endpoint.replace('/', ' ').lstrip().lower() + ' .'
    vocab_sentences.append(sentence)

id_to_action = {i:a for i, a in enumerate(vocab_sentences)}

# %%
# convert the tokenized user actions during online banking to API endpoint calls
actions_raw = df['actions'].to_list()

actions = []
for action in actions_raw:

    action_str = (action.replace('[', '')
           .replace(']', '')
           .replace(' ', '')
           .split(','))
    
    action_ids = []
    for id in action_str:
        if id:
            
            action_ids.append(id_to_action[int(id)])

    actions.append(' '.join(action_ids))

# %%
# plot the words to see most frequent
word_list = ' '.join(actions).replace(' . ', ' ').split()
fdist = FreqDist(word_list[2:])
fdist.plot(30)

# %%
# create wordcloud to visualize popular words
wordcloud = WordCloud().generate(' '.join(word_list))

plt.figure(figsize = (12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# %%
sequence_length = [len(s.split()) for s in actions]

# %%
fig = plt.hist(sequence_length, bins = 50, edgecolor = 'black')
plt.title('Token Length Distribution')
plt.xlabel('Token Length')
plt.ylabel('Number of Sentences')

# %%
# create an array of labels
labels = df['is_fraud'].to_list()

print(f'there are {sum(labels)} fraudulent transactions')
print(f'which is only {sum(labels)/len(labels)*100:0.2f}% of the total transactions')

# %%
# seperate the data into training and testing datasets
# enable the stratify option to ensure there are proportional amounts of fraudulent transactions in the training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(actions, labels, test_size=0.2, shuffle=True, stratify=labels)

# %%
print(sum(y_train))
print(sum(y_test))

# %% [markdown]
# ## Deepseek R1

# %% [markdown]
# Load in the necessary dependencies:

# %%
# !pip install -r requirements.txt

# %%
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# %% [markdown]
# ### Define the dataset
# We want to train on uniform length of data per each index of data, so we set `padding` and `max_length` to be the same. We also turn on `truncation`. 

# %%
class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# %% [markdown]
# ### Define the model
# Lightning framework encompasses training, validation and optimzations all into a single class.
# 
# We'll use a learning rate of `2e-5` because this is very common for large language models to guard against `catastrophic forgetting` [TODO: Reference]. 
# We'll setup the fine tuning process to add in a new linear layer at the end that takes the final hidden representation and maps it to logits for binary classification.
# We'll use `CrossEntropyLoss` as the loss function because we are doing a classification problem that outputs two classes and `CrossEntropyLoss` has shown to be a good industry standard for loss functions for classiciation problems (TODO: reference)
# For PyTorch Lightining, we also have to define a `configure_optimizers` method. For this DeepSeek model, we are using the `AdamW` optimizer because it is also a common optimier instead of things like SGD and RMS (todo: find refernces)

# %%
from transformers import AutoModel

class DeepSeekClassifier(pl.LightningModule):
    def __init__(self, model_name, lr=2e-5, n_classes=2):
        super().__init__()
        self.save_hyperparameters()
        hf_token = os.getenv("HF_TOKEN")
        # self.base_model = AutoModel.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.float16,
        #     trust_remote_code=True,
        #     token=hf_token,
        #     device_map="auto",
        #     low_cpu_mem_usage=True
        # )

        self.base_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=hf_token
        )

        self.classifier = nn.Linear(self.base_model.config.hidden_size, n_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids = input_ids.to(self.base_model.device)
        attention_mask = attention_mask.to(self.base_model.device)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1] 
        cls_rep = last_hidden[:, -1, :].to(self.classifier.weight.dtype) 
        logits = self.classifier(cls_rep)
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fn(logits, labels)
        else:
            loss = None
        return {"loss": loss, "logits": logits}

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        self.log("train_loss", out["loss"])
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        preds = torch.argmax(out["logits"], dim=1)
        labels = batch["labels"]
        acc = accuracy_score(labels.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.cpu(), preds.cpu(), average="binary"
        )
        self.log_dict({"val_acc": acc, "val_precision": precision, "val_recall": recall, "val_f1": f1})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# %% [markdown]
# ### Define training params
# We'll start with using the smallest Deepseek R1 model available, which is the `R1 Zero`. For length of tokens, lets take a quick look at the data:

# %%
found_legit = False
found_fraud = False

for x, y in zip(X_train, y_train):
    if y == 0 and not found_legit:
        print("Legit example (label 0):")
        print(x)
        found_legit = True
    elif y == 1 and not found_fraud:
        print("\nFraud example (label 1):")
        print(x)
        found_fraud = True

    if found_legit and found_fraud:
        break


# %% [markdown]
# It looks like legit examples are much smaller in the above case so we want to check the average length of all transcations

# %%
from collections import defaultdict
import numpy as np

def compute_avg_lengths(X, y):
    lengths_by_label = defaultdict(list)

    for x, label in zip(X, y):
        token_count = len(x.split())
        lengths_by_label[label].append(token_count)

    avg_lengths = {}
    for label in sorted(lengths_by_label):
        avg = np.mean(lengths_by_label[label])
        avg_lengths[label] = avg
        label_name = 'Fraudulent' if label == 1 else 'Legit'
        print(f"Average token length for {label_name} transactions (label {label}): {avg:.2f}")

    return avg_lengths

print("== X_train ==")
train_avg_lengths = compute_avg_lengths(X_train, y_train)

print("\n== X_test ==")
test_avg_lengths = compute_avg_lengths(X_test, y_test)


# %%
from collections import defaultdict
import numpy as np

def compute_avg_lengths(X, y):
    lengths_by_label = defaultdict(list)

    for x, label in zip(X, y):
        token_count = len(x.split())
        lengths_by_label[label].append(token_count)

    avg_lengths = {}
    for label in sorted(lengths_by_label):
        avg = np.mean(lengths_by_label[label])
        avg_lengths[label] = avg
        label_name = 'Fraudulent' if label == 1 else 'Legit'
        print(f"Average token length for {label_name} transactions (label {label}): {avg:.2f}")

    return avg_lengths

print("== X_train ==")
train_avg_lengths = compute_avg_lengths(X_train, y_train)

print("\n== X_test ==")
test_avg_lengths = compute_avg_lengths(X_test, y_test)


# %% [markdown]
# Since we know that average token length for fraud transactions are larger, we need to ensure that our training parameters account for this, thus we will pick `MAX_LEN` of `512`. We will do small training batches since the data is larger, so lets start with `2`. For now, lets start with training for `3` epochs.

# %%
MODEL_NAME = "deepseek-ai/DeepSeek-Coder-1.3B-base"
MAX_LEN = 512
BATCH_SIZE = 2
LR = 2e-5
N_EPOCHS = 3

# %% [markdown]
# Lets start by training on some dummy data to make sure our trainer is functioning correctly:

# %%
X_train_dummy = [
    "profile getcustomerresponse . campaign getbalance . accounts accounts_full . profile userprofile . loans list . transactions series post . templates meta . taxfree gettaxgoal . authentication fastlogin .",  # legit
    "profile getcustomerresponse . corporatemanagement cloneuser . verification verifycode . transactions series post . billpayments pay . p2b getposinfo . api accounts accounts_full ."  # fraud
]
y_train_dummy = [0, 1]

X_test_dummy = [
    "campaign getbalance . profile userprofile . accounts accounts_full . loans list . taxfree gettaxgoal ."
]
y_test_dummy = [0]


# %% [markdown]
# Now lets do the training and save the model:

# %%
def train_model(X_train, y_train, X_test, y_test, save_path="trained_model"):
    torch.set_float32_matmul_precision("medium")
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = FraudDataset(X_train, y_train, tokenizer, MAX_LEN)
    test_dataset = FraudDataset(X_test, y_test, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)

    model = DeepSeekClassifier(model_name=MODEL_NAME, lr=LR)

    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

    # Save the model and tokenizer after training
    model.base_model.save_pretrained(save_path)
    model.classifier.cpu()  # move classifier to CPU before saving
    torch.save(model.classifier.state_dict(), os.path.join(save_path, "classifier_head.pt"))
    tokenizer.save_pretrained(save_path)


# %% [markdown]
# Train using the dummy data to make sure we are good:

# %%
train_model(X_train_dummy, y_train_dummy, X_test_dummy, y_test_dummy, save_path="trained_model_dummy")

# %% [markdown]
# Now do it for real:

# %%
print("Training the real model:")
train_model(X_train, y_train, X_test, y_test, save_path="trained_model_real")

# %% [markdown]
# ### Methodology
# 
# 1. Explan your Deep Learning process / methodology
# 
# 
# 
# 2. Introduce the Deep Neural Networks you used in your project
#  * Model 1
#     * Description 
#  
#  * Model 2
#     * Description
#  
#  * Ensemble method
#      * Description 
#  
#  
# 3. Add keywords  
# **Keywords:** natural language processing, sentiment analysis, clustering, binary classification, multi-label classification, prediction
# 	___
#  **Example**
# * ConvNet
#     * A convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery(source Wikipedia). 
#  
# * **Keywords:** supervised learning, classification, ...

# %%
# TODO: Add code

# %% [markdown]
# ### Model Fitting and Validation
# 
# 1. model 1 
#     - decription 
# 2. model 2
#     - decription 

# %%
# TODO: Add Code

# %% [markdown]
# ### Model Evaluation 
# 
# * Examine your models (coefficients, parameters, errors, etc...)
# 
# * Compute and interpret your results in terms of accuracy, precision, recall, ROC etc. 

# %%
# TODO: Add code

# %% [markdown]
# ### Issues / Improvements
# 1. Dataset is very small
# 2. Use regularization / initialization
# 3. Use cross-validaiton
# 4. ...

# %% [markdown]
# ###  References
#    - Academic (if any)
#    - Online (if any)
# 	

# %% [markdown]
# ### Credits
# 
# - If you use and/or adapt your code from existing projects, you must provide links and acknowldge the authors. Keep in mind that all documents in your projects and code will be check against the official plagiarism detection tool used by Penn State ([Turnitin](https://turnitin.psu.edu))
# 
# > *This code is based on .... (if any)*


