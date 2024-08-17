
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import spacy
import logging
import nltk
nltk.download('punkt',quiet=True)
import click
from nltk.tokenize import sent_tokenize

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
)

import json
nlp = spacy.load('en_coreference_web_trf')

def resolve_references(sent):

    doc = nlp(sent)
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    for cluster in clusters:
        first_mention = cluster[0]
        for mention_span in list(cluster)[1:]:
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_

            for token in mention_span[1:]:
                token_mention_mapper[token.idx] = ""

    for token in doc:
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        else:
            output_string += token.text + token.whitespace_

    return output_string

def preprocess(file_name):
    logging.info("Preprocessing ...")
    with open(file_name, 'r+') as f:
        content = f.read().replace('*', '').replace('#', '').replace('-', '').replace('\xa0', ' ')

    paragraphs = content.split('\n\n')
    coref_resolved = [resolve_references(k) for k in paragraphs]

    preprocessed_lines = []
    for p in coref_resolved:
        preprocessed_lines.extend(p.split('\n'))
    sents = []
    for p in preprocessed_lines:
        sents.extend(sent_tokenize(p))

    return sents

@click.command()
@click.option('--policy_doc', default='privacy_hotcrp.md',
              help='High-level requirement specification document',
              show_default=True,
              required=True,
              )
def main(policy_doc):
    print("\n ============================= Starting AGentV =============================\n")
    logging.info(f"Policy document: {policy_doc}")
    sents = preprocess(policy_doc)
    with open('high_level_requirements.json', 'w') as f:
        json.dump(sents, f)
    logging.info("Preprocessing is completed!")


if __name__ == '__main__':
    main()