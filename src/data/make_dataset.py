# -*- coding: utf-8 -*-
import os
import re
import pandas
import spacy
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from gensim.models.phrases import Phrases, Phraser

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

def convert_raw_corpus_to_df(path_to_corpus, clean = False):
    """ Converts a raw corpus to a pandas dataframe. 
    
    The function assumes that the raw corpus is a directory with every document
    represented as a separate file within. It returns a dataframe in which
    every row corresponds to a single sentence in the corpus. The dataframe
    contains columns 'document_index' and 'sentence_index' to uniquely identify
    each sentence, as well as the column 'date' which corresponds to the date
    when the document was published. If the flag <clean> is set to true, the sentences 
    in the data frame
        
    Args:
        path_to_corpus (str): path to the directory containing the corpus
        clean (bool): flag whether to clean the sentence
    
    Returns:
        pandas.DataFrame: containing the corpus subdivided into sentences 

    """

    def is_accepted(token):
        output = token.is_punct
        output |= token.is_space
        output |= token.is_stop
        return not output 

    corpus = RawCorpusIter(path_to_corpus)

    nlp = spacy.load("es_core_news_sm", disable=['ner'])
    # TODO: Spacy v2 does not use POS tags when lemmatizing Spanish, this
    # should be addressed in v3

    raw_sentences = []
    cleaned_sentences = []
    sentence_indices = []
    document_indices = []
    dates = []
    for document_id, (date, text) in enumerate(corpus):
        # In the Zapatista corpus hypthen is often used to indicated
        # speach instead of n-dash, but spacy does not recognise it as
        # punctuation. Replace hypthen at the start of a sentence by an n-dash
        text = re.sub('^-', '–', text)  
        text = re.sub('(\s)-', '\g<1>–', text)
        doc = nlp(text)
        sentence_index = 0
        for sentence in doc.sents:
            raw_sentence = sentence.text.strip()  
            if raw_sentence != "":
                raw_sentences.append(raw_sentence)
                sentence_indices.append(sentence_index)
                sentence_index+=1
                if clean:
                    filtered_words = [token.lemma_.lower() for token in sentence if is_accepted(token)]    
                    cleaned_sentences.append(" ".join(filtered_words))
        document_indices += [document_id] * sentence_index
        dates += [pandas.to_datetime(date[:10])] * sentence_index
    
    if clean:
        df = pandas.DataFrame({'date':dates,
                               'document_index': document_indices,
                               'sentence_index': sentence_indices,
                               'raw_sentence': raw_sentences,
                               'cleaned_sentence': cleaned_sentences })

    else:
        df = pandas.DataFrame({'date':dates,
                               'document_index': document_indices,
                               'sentence_index': sentence_indices,
                               'raw_sentence': raw_sentences,
                               'date':dates})

    df = df.dropna().reset_index(drop=True)
    return df 

def combine_phrases_in_corpus(corpus_df, column, min_count=75):
    """ Use gensim phrases to find phrases in the corpus.

    Args:
        corpus_df (pandas.DataFrame): the input corpus 
        column (str): the name of the column to be processed
        min_count (int): min_count hyperparameter of gensim module
    """
    try:
        sentences = [sent.split() for sent in corpus_df[column]]
    except KeyError:
        print(f"{column} is not a name of a column of the input dataframe")

    phrases_model = Phrases(sentences, min_count=min_count, progress_per=10000)
    phrases_model = Phraser(phrases_model)

    sentences_with_phrases = [" ".join(sent) for sent in phrases_model[sentences]]

    corpus_df[column + "_with_phrases"] = sentences_with_phrases


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--clean/--no-clean', default=False, help="Make the senteces lower case and remove punctuation")
@click.option('--phrases/--no-phrases', default=False, help="Combine commonly used phrases into a single token")
def main(input_filepath, output_filepath, clean, phrases):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = convert_raw_corpus_to_df(input_filepath, clean)
    if clean and phrases:
        combine_phrases_in_corpus(df, 'cleaned_sentence')
    df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
