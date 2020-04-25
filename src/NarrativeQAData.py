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
from cdqa.reader import BertProcessor, BertQA

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


def ask_doc_qs_cdqa(docdf, cdqa_pipeline, qs):
    '''
    Given docdf, a pandas dataframe representing a document 
    or group of documents with columns 'title' and 'paragraphs', 
    a joblib model, and a list of questions, this function
    makes cdqa queries for each question and prints the predicted 
    answers.
    '''
    cdqa_pipeline.fit_retriever(df=docdf)
    for q in qs:
        prediction = cdqa_pipeline.predict(q)
        print('question: {}'.format(q))
        print('predicted answer: {}'.format(prediction[0]))
        #print('title: {}'.format(prediction[1]))
        #print('paragraph: {}'.format(prediction[2]))
        print("\n\n")
        
        

def pred(qdf, storiesDir, model):
    '''
    Given a narrativeqa dataframe, a directory where the narrative 
    stories are located, and a joblib model, this function asks the questions 
    corresponding to each document using the ask_doc_qs function above. 
    '''
    docIds = qdf['document_id'].unique()
    cdqa_pipeline = QAPipeline(reader=model, min_df=1, max_df=5.0)
    for docId in docIds[:1]:
        docEntries = qdf.loc[qdf['document_id'] == docId]
        docFile = os.path.join(storiesDir, docId + ".content")
        qList = docEntries["question"].tolist()
        with open(docFile, "r") as doc:
            data = doc.read()
            pars = data.split("\n\n\n")
            docdf = pd.DataFrame({'title' : [docId], "paragraphs" : [pars]})
            ask_doc_qs_cdqa(docdf, cdqa_pipeline, qList)
            #for q in qList:
            #    Drqa_process(q)


def narrative_df_to_squadlike_txt(df, outName):
    '''
    Convert the dataframe representing the set of NarrativeQA
    questions into the SQUaD-like txt file format expected by DrQA.
    '''
    df['answer'] = df[["answer1","answer2"]].values.tolist()
    Json = df[["question", "answer"]].to_json(orient="records", lines=True)
    with open(outName, 'w') as outfile:
        outfile.write(Json)

def train_cdqa_reader(squadlikeJson, outModel, inModel=None, negatives=False):
    '''Train a cdqa model on a dataset formatted like squad jsons'''

    # Currently this causes GPU memory errors if run
    # When running on the CPU it is much too slow ~73 hrs for SQuAD 2.0
    # Looking into ways to fix it. 
    
    # Fine tune an existing model:
    if inModel:
        cdqa_pipeline = QAPipeline(reader=inModel)
        cdqa_pipeline.fit_reader(squadlikeJson)
    # train a model from scratch:
    else:
        train_processor = BertProcessor(do_lower_case=True, is_training=True, version_2_with_negative=negatives)
        train_examples, train_features = train_processor.fit_transform(X=squadlikeJson)
        reader = BertQA(train_batch_size=1,
                        learning_rate=3e-5,
                        num_train_epochs=2,
                        do_lower_case=True,
                        output_dir=os.path.dirname(outModel))

        reader.fit(X=(train_examples, train_features))
        reader.model.to('cpu')
        reader.device = torch.device('cpu')
    # Output model:
    joblib.dump(reader, os.path.join(reader.output_dir, outModel))

def narrative_df_to_squadlike_json(df, outName):
    ''' 
    Convert narrativeQA data to squadlike json format. Left blank for now because it is unclear how to 
    handle the paragraph-question correspondence for the moment. 
    '''
    pass
    
        

if __name__ == "__main__":
    # Define file and directory paths:
    qcsv = './data/narrativeqa/qaps.csv'
    storiesDir = './data/narrativeqa/stories'
    inModel = './models/bert_qa.joblib'
    # Split data into three sets:
    qsTrain, qsVal, qsTest = split_train_test_val(qcsv)

    # Train a cdqa model on squad 2:
    SquadTrainJson = './data/SQuAD/SQuAD-v2.0-train.json'
    outModel = './models/bert_qa_squad_1_and_2.joblib'
    #train_cdqa_reader(SquadTrainJson, outModel, negatives=True)

    # Convert narrativeqa data into the squad-like txts for DrQA eval:
    #narrative_df_to_squadlike_txt(qsTrain, './data/narrativeqa/train.txt')
    #narrative_df_to_squadlike_txt(qsVal, './data/narrativeqa/val.txt')
    #narrative_df_to_squadlike_txt(qsTest, './data/narrativeqa/test.txt')

    # Run cdqa predicitons on narrativeqa training data:
    # This can probably easily be repurposed to train the
    # cdqa retriever (but probably not the reader). 
    pred(qsTrain, storiesDir, inModel)


