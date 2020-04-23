import os
import sys
import pandas as pd
from ast import literal_eval

from cdqa.utils.converters import pdf_converter
from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model

def split_train_test_val(qcsv):
    qs = pd.read_csv(qcsv)
    qsTrain = qs.loc[qs['set'] == 'train']
    qsVal = qs.loc[qs['set'] == 'valid']
    qsTest = qs.loc[qs['set'] == 'test']
    return qsTrain, qsVal, qsTest


def ask_doc_qs(docdf, model, qs):
    print(docdf)
    cdqa_pipeline = QAPipeline(reader=model, min_df=1, max_df=5.0)
    cdqa_pipeline.fit_retriever(df=docdf)
    for q in qs:
        prediction = cdqa_pipeline.predict(q)
        print('question: {}'.format(q))
        print('predicted answer: {}'.format(prediction[0]))
        #print('title: {}'.format(prediction[1]))
        #print('paragraph: {}'.format(prediction[2]))
        


def pred(qdf, storiesDir, model):
    docIds = qdf['document_id'].unique()
    for docId in docIds[:1]:
        docEntries = qdf.loc[qdf['document_id'] == docId]
        docFile = os.path.join(storiesDir, docId + ".content")
        qList = docEntries["question"].tolist()
        with open(docFile, "r") as doc:
            pars = doc.read()
            docdf = pd.DataFrame({'title' : [docId], "paragraphs" : [[pars]]})
            ask_doc_qs(docdf, model, qList)
        


if __name__ == "__main__":
    qcsv = './data/narrativeqa/qaps.csv'
    storiesDir = './data/narrativeqa/stories'
    model = './models/bert_qa.joblib'
    qsTrain, qsVal, qsTest = split_train_test_val(qcsv)
    pred(qsTrain, storiesDir, model)


