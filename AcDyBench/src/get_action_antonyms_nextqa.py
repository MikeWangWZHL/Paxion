import json
import os
from collections import defaultdict
from nltk.corpus import wordnet
import nltk
import spacy
import pyinflect
from tqdm import tqdm
from typing import List, Dict, Any, Mapping
import random
import logging
import re
import pandas as pd

def get_action_antonym_single_text(
    original_clip_text,
    nlp, # spacy model
    ignored_verbs = {},
    additional_antonyms_map: Dict[str,str] = {}
):
    mapped_verbs = set()
    unmapped_verbs = set()

    # process with spacy
    doc = nlp(original_clip_text)

    antonym_clip_text = original_clip_text
    for token in doc:
        lower_text = token.text.lower()

        # Check if the token is a verb and has antonyms
        if token.pos_ == "VERB":
            # Get the first antonym of the verb using WordNet
            antonym = token.text

            # try find it in the manual mapping table
            key = antonym.lower()
            if key in additional_antonyms_map:
                antonym = additional_antonyms_map[key]

            # if nothing found, check with wordnet
            if wordnet.synsets(token.text, pos=wordnet.VERB) and antonym == token.text:
                for syn in wordnet.synsets(token.text, pos=wordnet.VERB):
                    for lemma in syn.lemmas():
                        if lemma.antonyms():
                            antonym = lemma.antonyms()[0].name()
                            break
                    if antonym != token.text:
                        break

            # replace token if antonym is found and it is not in the IGNORED_VERBS list
            if antonym != token.text and token.text not in ignored_verbs:

                antonym_verb = antonym.replace("_"," ") # antonym from wordnet can be like "switch_on"

                # Match the tense of the antonym to the original verb
                antonym_verb_tokens = [t for t in nlp(antonym_verb)]
                if len(antonym_verb_tokens) == 1:
                    antonym = antonym_verb_tokens[0]._.inflect(token.tag_, inflect_oov=True)
                else:
                    antonym_verb_first_token = antonym_verb_tokens[0]._.inflect(token.tag_, inflect_oov=True)
                    antonym = antonym_verb_first_token + ' ' + ' '.join([t.text for t in antonym_verb_tokens[1:]])
                if antonym is None:
                    antonym = antonym_verb
                    print(f"WARNING: tense matching error! back to original: {token.text} -> {antonym}")

                antonym_clip_text = antonym_clip_text.replace(token.text, antonym)
                # print(f"VERB: {token.text} -> {antonym}")

                if lower_text not in mapped_verbs:
                    # logging.info(f'VERB: {lower_text} --> {antonym}')
                    mapped_verbs.add(lower_text)
            else:
                if lower_text not in unmapped_verbs:
                    # logging.info(f'Failed to map verb: {lower_text}')
                    unmapped_verbs.add(lower_text)

        # change the ADPs such as "up" -> "down"
        elif token.pos_ == "ADP" and wordnet.synsets(token.text, pos=wordnet.ADV):
            # Get the first antonym of the particle using WordNet
            antonym = token.text

            for syn in wordnet.synsets(token.text, pos=wordnet.ADV):
                for lemma in syn.lemmas():
                    if lemma.antonyms():
                        antonym = lemma.antonyms()[0].name()
                        break
                if antonym != token.text:
                    break
            # replace token if antonym is found
            if antonym != token.text:
                antonym_clip_text = antonym_clip_text.replace(token.text, antonym)
                # print(f"ADP: {token.text} -> {antonym}")

                if lower_text not in mapped_verbs:
                    # logging.info(f'ADP: {lower_text} --> {antonym}')
                    mapped_verbs.add(lower_text)
            else:
                if lower_text not in unmapped_verbs:
                    # logging.info(f'Failed to map adposition: {lower_text}')
                    unmapped_verbs.add(lower_text)
    
    return antonym_clip_text

def get_action_antonyms_nextqa(output_ann_path=None):
    ann_path = '../../datasets/NextQA/nextqa'
    assert output_ann_path is not None # set the output path
    os.makedirs(output_ann_path, exist_ok=True)
    # reusing the ignored verbs and additional antonym mapping from ssv2 and ego4d
    additional_antonyms_map = json.load(open("additional_antonyms_mapping_ego4d.json"))
    ignored_verbs = json.load(open("ignored_verbs_ego4d.json"))
    additional_antonyms_map.update(json.load(open("additional_antonyms_mapping_ssv2.json")))
    ignored_verbs += json.load(open("ignored_verbs_ssv2.json"))

    print(additional_antonyms_map)
    print(ignored_verbs)

    # set up spacy
    nlp = spacy.load("en_core_web_trf", disable=["ner", "textcat"]) # this one is slower but with higher accurarcy

    # Start mapping each stage
    for split in ['val', 'train', 'test']:
        csv = os.path.join(ann_path, f"{split}.csv")
        df = pd.read_csv(csv, header=0)
        
        # df = df.head(5)
        print()
        print("Split:", split)
        print("Length:", len(df))

        action_antonym_choices_all = []
        unmapped_text = set()
        for index, row in tqdm(df.iterrows(),total=len(df)):
            original_choices = [row[a] for a in ['a0','a1','a2','a3','a4']]
            # find antonym
            action_antonym_choices_cands = [
                get_action_antonym_single_text(t, nlp, ignored_verbs, additional_antonyms_map)
                for t in original_choices 
            ]
            assert len(original_choices) == len(action_antonym_choices_cands)
            action_antonym_choices = []
            for i in range(len(original_choices)):
                if action_antonym_choices_cands[i] != original_choices[i]:
                    action_antonym_choices.append(action_antonym_choices_cands[i])
                    # print(f"mapped:{original_choices[i]} -> {action_antonym_choices_cands[i]}")
                else:
                    unmapped_text.add(action_antonym_choices_cands[i])
                    # print("unmapped:", action_antonym_choices_cands[i])
            # add to column value
            action_antonym_choices_all.append(action_antonym_choices)
        df['action_antonym_choices'] = action_antonym_choices_all
        
        output_csv = os.path.join(output_ann_path, f"{split}.csv")
        df.to_csv(output_csv, index=False)

if __name__ == "__main__":

    random.seed(42)
    logging.basicConfig(level=logging.INFO)

    get_action_antonyms_nextqa(output_ann_path="<set up output path>")
    