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
        

def Drqa_process(question, candidates=None, top_n=1, n_docs=5):
    '''Make a DRQA question'''
    predictions = DrQA.process(
        question, candidates, top_n, n_docs, return_context=True
    )
    table = prettytable.PrettyTable(
        ['Rank', 'Answer', 'Doc', 'Answer Score', 'Doc Score']
    )
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p['span'], p['doc_id'],
                       '%.5g' % p['span_score'],
                       '%.5g' % p['doc_score']])
    print('Top Predictions:')
    print(table)
    print('\nContexts:')
    for p in predictions:
        text = p['context']['text']
        start = p['context']['start']
        end = p['context']['end']
        output = (text[:start] +
                  colored(text[start: end], 'green', attrs=['bold']) +
                  text[end:])
        print('[ Doc = %s ]' % p['doc_id'])
        print(output + '\n')
        

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



        


if __name__ == "__main__":
    drqa.tokenizers.set_default('corenlp_classpath', './data/corenlp')
    tok = drqa.tokenizers.CoreNLPTokenizer()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader-model', type=str, default=None,
                        help='Path to trained Document Reader model')
    parser.add_argument('--retriever-model', type=str, default=None,
                        help='Path to Document Retriever model (tfidf)')
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to Document DB')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help=("String option specifying tokenizer type to "
                              "use (e.g. 'corenlp')"))
    parser.add_argument('--candidate-file', type=str, default=None,
                        help=("List of candidates to restrict predictions to, "
                              "one candidate per line"))
    parser.add_argument('--no-cuda', action='store_true',
                        help="Use CPU only")
    parser.add_argument('--gpu', type=int, default=-1,
                        help="Specify GPU device id to use")
    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
    else:
        logger.info('Running on CPU only.')

    if args.candidate_file:
        logger.info('Loading candidates from %s' % args.candidate_file)
        candidates = set()
        with open(args.candidate_file) as f:
            for line in f:
                line = utils.normalize(line.strip()).lower()
                candidates.add(line)
        logger.info('Loaded %d candidates.' % len(candidates))
    else:
        candidates = None

    logger.info('Initializing DRQA pipeline...')
    DrQA = pipeline.DrQA(
    cuda=args.cuda,
    fixed_candidates=candidates,
    reader_model=args.reader_model,
    ranker_config={'options': {'tfidf_path': args.retriever_model}},
    db_config={'options': {'db_path': args.doc_db}},
    tokenizer=args.tokenizer
    )

    qcsv = './data/narrativeqa/qaps.csv'
    storiesDir = './data/narrativeqa/stories'
    model = './models/bert_qa.joblib'
    qsTrain, qsVal, qsTest = split_train_test_val(qcsv)
    pred(qsTrain, storiesDir, model)


