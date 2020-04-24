import os
import sys
import pandas as pd
from ast import literal_eval

import torch
import argparse
import code
import prettytable
import logging

from termcolor import colored
from drqa import pipeline
from drqa.retriever import utils
import drqa.tokenizers

from cdqa.utils.converters import pdf_converter
from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model

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


def ask_doc_qs_cdqa(docdf, model, qs):
    '''
    Given docdf, a pandas dataframe representing a document 
    or group of documents with columns 'title' and 'paragraphs', 
    a joblib model, and a list of questions, this function
    makes cdqa queries for each question and prints the predicted 
    answers.
    '''
    cdqa_pipeline = QAPipeline(reader=model, min_df=1, max_df=5.0)
    cdqa_pipeline.fit_retriever(df=docdf)
    for q in qs:
        prediction = cdqa_pipeline.predict(q)
        print('question: {}'.format(q))
        print('predicted answer: {}'.format(prediction[0]))
        print('title: {}'.format(prediction[1]))
        print('paragraph: {}'.format(prediction[2]))
        print("\n\n")
        
        

def pred(qdf, storiesDir, model):
    '''
    Given a narrativeqa dataframe, a directory where the narrative 
    stories are located, and a joblib model, this function asks the questions 
    corresponding to each document using the ask_doc_qs function above. 
    '''
    docIds = qdf['document_id'].unique()
    for docId in docIds[:1]:
        docEntries = qdf.loc[qdf['document_id'] == docId]
        docFile = os.path.join(storiesDir, docId + ".content")
        qList = docEntries["question"].tolist()
        with open(docFile, "r") as doc:
            data = doc.read()
            pars = data.split("\n\n\n")
            docdf = pd.DataFrame({'title' : [docId], "paragraphs" : [pars]})
            ask_doc_qs_cdqa(docdf, model, qList)
            for q in qList:
                Drqa_process(q)


def narrative_df_to_squadlike(df, outName):
    '''
    Convert the dataframe representing the set of NarrativeQA
    questions into the SQUaD-like txt file format expected by DrQA.
    '''
    df['answer'] = df[["answer1","answer2"]].values.tolist()
    Json = df[["question", "answer"]].to_json(orient="records", lines=True)
    with open(outName, 'w') as outfile:
        outfile.write(Json)
    
        

if __name__ == "__main__":
    qcsv = './data/narrativeqa/qaps.csv'
    storiesDir = './data/narrativeqa/stories'
    model = './models/bert_qa.joblib'
    qsTrain, qsVal, qsTest = split_train_test_val(qcsv)
    
    narrative_df_to_squadlike(qsTrain, './data/narrativeqa/train.txt')
    narrative_df_to_squadlike(qsVal, './data/narrativeqa/val.txt')
    narrative_df_to_squadlike(qsTest, './data/narrativeqa/test.txt')
    
    pred(qsTrain, storiesDir, model)


