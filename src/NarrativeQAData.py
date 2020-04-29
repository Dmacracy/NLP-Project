import os
import sys
import pandas as pd
import numpy as np
import scipy.sparse as sp
from ast import literal_eval

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

import argparse
import code
import prettytable
import logging
import transformers

from gutenberg.cleanup import strip_headers



def split_train_test_val(qcsv):
    '''
    Splits narrative QA dataset 
    into three dataframes: train, validation, and test.
    '''
    qs = pd.read_csv(qcsv)
    qsTrain = qs.loc[qs['set'] == 'train']
    qsVal = qs.loc[qs['set'] == 'valid']
    qsTest = qs.loc[qs['set'] == 'test']
    return qsTrain, qsVal, qsTest

class B25mScorer:
    '''
    A B25m document retreival scorer.
    See https://en.wikipedia.org/wiki/Okapi_BM25
    '''
    def __init__(self, docs, k=1.6, b=0.75):
        self.N = len(docs) # num docs
        self.Ds = np.array([len(d) for d in docs]) # length of each doc
        self.avgD = np.mean(self.Ds) # avg doc length
        self.docs = [tokenizer.tokenize(doc) for doc in docs] # tokenized docs
        self.k = k # k and b params.
        self.b = b

    def score(self, q):
        # for a given query, return a list of scores for each doc
        scores = []
        q_tokenized = tokenizer.tokenize(q)
        for idx, doc in enumerate(self.docs):   
            score = 0
            for q_tok in q_tokenized:
                # number of docs containing q token:
                n_q = np.sum([int(q_tok in d) for d in self.docs])
                # inverse doc freq for q word in this doc:
                idf = np.log((self.N - n_q + 0.5) / (n_q + 0.5))
                # term freq:
                tf = doc.count(q_tok) / self.Ds[idx]
                numerator = tf * (self.k + 1)
                denominator = tf + self.k * (1 - self.b + (self.b * self.Ds[idx] /  self.avgD))
                # sum the doc's score over all query tokens
                score += idf * (numerator / denominator)
            # add doc's score to list:
            scores.append(score)
        return np.array(scores)
    
    def get_best_n_doc_idxs(self, q, n):
        # return the document indices of the highest n scoring docs for a given 
        scores = self.score(q)
        best_n_idxs = np.argsort(-scores)[:n]
        print(best_n_idxs)
        return best_n_idxs
    

def answer_question(question, textids, texts):
    answers = []
    for textid in textids:
        text = texts[textid]
        print("Question:\n", question)
        print("Text:\n", text)
        inputs = tokenizer.encode_plus(question, text, return_tensors="pt")
        # Need to pop token type ids when using distilbert because this model does not 
        # handle them, but the encoder still sets them for some reason. 
        inputs.pop('token_type_ids', None)
        input_ids = inputs["input_ids"].tolist()[0]
    
        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = model(**inputs)
    
        # Get the most likely beginning of answer with the argmax of the score
        answer_start = torch.argmax(answer_start_scores)
        # Get the most likely end of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        
        if "answer" == '[CLS]':
            print(f"Answer: I cannot answer that.\n")
        else:
            print(f"Answer: {answer}\n")
        answers.append(answer)
    return answers

    
def pred(qdf, storiesDir, model, predOutDir, num_best_pars=1):
    '''
    Given a narrativeqa dataframe, a directory where the narrative 
    stories are located, and a pytorch model, this function asks the questions 
    corresponding to each document. 
    '''
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    model = AutoModelForQuestionAnswering.from_pretrained(modelPath)
    
    docIds = qdf['document_id'].unique()
    for docId in docIds:
        docEntries = qdf.loc[qdf['document_id'] == docId]
        docFile = os.path.join(storiesDir, docId + ".content")
        with open(docFile, "r") as doc:
            try:
                data = doc.read()
                if '<html>' not in data:
                    data = strip_headers(data)
                    pars = data.split("\n\n")
                    BMScorer = B25mScorer(pars)
                    docEntries["best_ranked_pars"] = docEntries["question"].apply(lambda q : BMScorer.get_best_n_doc_idxs(q, num_best_pars))
                    docEntries['predicted_answers'] = docEntries[["question", "best_ranked_pars"]].apply(lambda x : answer_question(*x, pars), axis=1)

                    docEntries.to_csv(os.path.join(predOutDir, docId + '.csv'), index=False)
            except UnicodeDecodeError:
                pass


def narrative_df_to_squadlike_txt(df, outName):
    '''
    Convert the dataframe representing the set of NarrativeQA
    questions into the SQUaD-like txt file format expected by DrQA.
    '''
    df['answer'] = df[["answer1","answer2"]].values.tolist()
    Json = df[["question", "answer"]].to_json(orient="records", lines=True)
    with open(outName, 'w') as outfile:
        outfile.write(Json)

if __name__ == "__main__":
    # Define file and directory paths:
    qcsv = '../data/narrativeqa/qaps.csv'
    storiesDir = '../data/narrativeqa/stories'
    modelPath = "../models/SQuAD2_trained_model"

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    model = AutoModelForQuestionAnswering.from_pretrained(modelPath)
    
    # Split data into three sets:
    qsTrain, qsVal, qsTest = split_train_test_val(qcsv)


    # Convert narrativeqa data into the squad-like txts for eval:
    #narrative_df_to_squadlike_txt(qsTrain, './data/narrativeqa/train.txt')
    #narrative_df_to_squadlike_txt(qsVal, './data/narrativeqa/val.txt')
    #narrative_df_to_squadlike_txt(qsTest, './data/narrativeqa/test.txt')


    predOutDir = '../data/predictions/'
    pred(qsTrain, storiesDir, modelPath, predOutDir, 4)


