#import drqa.tokenizers
#drqa.tokenizers.set_default('corenlp_classpath', './data/corenlp')
#tok = drqa.tokenizers.CoreNLPTokenizer()
#print(tok.tokenize('hello world').words())

import os
import sys
import pandas as pd
from ast import literal_eval

from cdqa.utils.converters import pdf_converter
from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model


if __name__ == "__main__":
    cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib', max_df=1.0)
    df = pdf_converter(directory_path='./data/pdf/')
    df.head()
    # Fit Retriever to documents
    cdqa_pipeline.fit_retriever(df=df)
    query = sys.argv[1]
    prediction = cdqa_pipeline.predict(query, return_all_preds=True)
    print('query: {}'.format(query))
    for pred in prediction:
        print(pred)
    #print('answer: {}'.format(prediction[0]))
    #print('title: {}'.format(prediction[1]))
    #print('paragraph: {}'.format(prediction[2]))

