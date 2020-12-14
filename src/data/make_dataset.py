# -*- coding: utf-8 -*-
import os
import pandas
import spacy
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv



class RawCorpusIter:
    """ An iterator of a raw corpus.
    
    The iterator assumes that the raw corpus is a directory with every document
    represented as a separate file within.
        
    Args:
        path_to_corpus (str): path to the directory containing the corpus
    
    Returns:
        iterator: when iterated this iterator returns a pair containing the name of
            the file and the content of the file 
    
    """
    def __init__(self, path_to_corpus):
        self.corpus_dir = path_to_corpus
        self.files = os.listdir(self.corpus_dir)

    def __iter__(self):
        self.file_num = 0
        return self

    def __next__(self):
        if self.file_num < len(self.files):
            n = self.file_num
            self.file_num += 1
            with open(os.path.join(self.corpus_dir, self.files[n]), "r") as file:
                return(self.files[n], file.read())
        else:
            raise StopIteration

def convert_raw_corpus_to_df(path_to_corpus):
    """ Converts a raw corpus to a pandas dataframe. 
    
    The function assumes that the raw corpus is a directory with every document
    represented as a separate file within. It returns a dataframe in which
    every row corresponds to a single sentence in the corpus. The dataframe
    contains columns 'document_index' and 'sentence_index' to uniquely identify
    each sentence, as well as the column 'date' which corresponds to the date
    when the document was published.
        
    Args:
        path_to_corpus (str): path to the directory containing the corpus
    
    Returns:
        pandas.DataFrame: containing the corpus subdivided into sentences 
    
    """
    corpus = RawCorpusIter(path_to_corpus)

    nlp = spacy.load("es_core_news_sm", disable=['ner', 'parser'])
    sentencizer = spacy.pipeline.Sentencizer()
    nlp.add_pipe(sentencizer)

    sentences = []
    sentence_indices = []
    document_indices = []
    dates = []
    for document_id, (date, text) in enumerate(corpus):
        doc = nlp(text)
        sentence_index = 0
        for sentence in doc.sents:
            stripped_sentence = sentence.text.strip()
            if stripped_sentence != "":
                sentences.append(stripped_sentence)
                sentence_indices.append(sentence_index)
                sentence_index+=1
        document_indices += [document_id] * sentence_index
        dates += [pandas.to_datetime(date[:10])] * sentence_index
    
    return pandas.DataFrame({'document_index':document_indices, 'sentence_index': sentence_indices, 'sentence':sentences, 'date':dates})

    return df

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = convert_raw_corpus_to_df(input_filepath)
    df.to_csv(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
