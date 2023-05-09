# %%
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


### === main function for getting action antonyms === ###
def get_action_antonyms(
    annotations: List[Dict[str,Any]],
    ignored_verbs = {},
    additional_antonyms_map: Dict[str,str] = {},
    sample_filter: Mapping[Dict,bool] = lambda x: True,
    text_key = 'clip_text'
):
    print('Ignored verbs:', ignored_verbs)

    # init spacy nlp pipeline
    # nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    nlp = spacy.load("en_core_web_trf", disable=["ner", "textcat"]) # this one is slower but with higher accurarcy

    mapped_verbs = set()
    unmapped_verbs = set()

    print('Total number of annotations:', len(annotations))
    annotations_with_action_antonym = []
    # filter out ann that has at least one action
    # for idx, sample in enumerate(annotation):
    for sample in tqdm(annotations):
        if sample_filter(sample):
            # load annotation text
            original_clip_text = sample[text_key]

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
                            logging.info(f'VERB: {lower_text} --> {antonym}')
                            mapped_verbs.add(lower_text)
                    else:
                        if lower_text not in unmapped_verbs:
                            logging.info(f'Failed to map verb: {lower_text}')
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
                            logging.info(f'ADP: {lower_text} --> {antonym}')
                            mapped_verbs.add(lower_text)
                    else:
                        if lower_text not in unmapped_verbs:
                            logging.info(f'Failed to map adposition: {lower_text}')
                            unmapped_verbs.add(lower_text)

            # if we successfully found and replaced verb with its antonym
            if original_clip_text != antonym_clip_text:
                sample["action_antonym_clip_text"] = antonym_clip_text
                annotations_with_action_antonym.append(sample)

    print('Number of annotations with action antonyms:', len(annotations_with_action_antonym))

    return annotations_with_action_antonym


### == Get antonyms for AcDybench-Ego4D == ###
def get_action_antonyms_ego4d(output_ann_path = None):
    
    def load_ego4d_annotation(dataset_name, vis_root, ann_path):
        if dataset_name == "egoclip_subset":
            # load annotation
            if isinstance(ann_path, list):
                ann_path = ann_path[0]
            clipuid_2_idx = defaultdict(list)
            ann_full = []
            with open(ann_path) as f:
                for line in f:
                    ann_instance = json.loads(line)
                    clip_uid = ann_instance['clip_uid']
                    # filter out missing video clips
                    if os.path.exists(os.path.join(vis_root, f'{clip_uid}.mp4')):
                        clipuid_2_idx[clip_uid].append(len(ann_full))
                        ann_full.append(ann_instance)
            annotation = ann_full
            print("full annotation size:", len(annotation))
            return annotation
        else:
            raise NotImplementedError

    vis_root = "../../datasets/Ego4D/video_clips/clips_downsampled_5fps_downsized_224x224"
    ann_path = "../ego4d/egoclip_subset_original.jsonl" # original annotation
    assert output_ann_path is not None # set output path

    # manual mapping of the verbs; this could be dataset-specific
    ADDITIONAL_ANTONYMS_MAPPING = json.load(open("additional_antonyms_mapping_ego4d.json"))

    # verbs that should be ignored; this could be dataset-specific
    IGNORED_VERBS = set(json.load(open("ignored_verbs_ego4d.json")))

    def is_valid_sample(sample: Dict[str,Any]):
        try:
            noun_idx = eval(sample['tag_noun']) # specific for egoclip annotation
            verb_idx = eval(sample['tag_verb']) # specific for egoclip annotation

            return len(verb_idx) >= 1 and len(noun_idx) >= 1
        except KeyError:
            return False

    # load annotation
    annotation = load_ego4d_annotation("egoclip_subset", vis_root, ann_path)
    annotations_with_action_antonym = get_action_antonyms(
        annotation,
        ignored_verbs=IGNORED_VERBS,
        additional_antonyms_map=ADDITIONAL_ANTONYMS_MAPPING,
        sample_filter=is_valid_sample
    )

    # output file
    with open(output_ann_path, 'w') as o:
        for line in annotations_with_action_antonym:
            o.write(json.dumps(line))
            o.write('\n')

### == Get antonyms for AcDybench-SSv2 == ###
def get_action_antonyms_ssv2(output_ann_path=None):
    vis_root = 'unused'
    ann_path = '../ssv2/original'
    assert output_ann_path is not None # set output path

    # load annotation
    import sys
    import os
    sys.path.append(os.path.realpath(os.path.join(__file__, '../../../src')))

    from data_new import PhysicalKnowledgeBenchDataset_SSv2

    # Load hand-mapped and ignored verbs
    additional_antonyms_map = json.load(open(os.path.realpath(os.path.join(__file__, '..', 'additional_antonyms_mapping_ssv2.json'))))
    ignored_verbs = json.load(open(os.path.realpath(os.path.join(__file__, '..', 'ignored_verbs_ssv2.json'))))

    # Start mapping each stage
    for stage in ['val', 'test', 'train']:
        ds = PhysicalKnowledgeBenchDataset_SSv2(
            vis_root,
            ann_path,
            stage,
            use_templates_as_labels=stage == 'test'
        )

        print(f'{stage} length before filtering by action antonyms: {len(ds.annotation)}')
        annotations_with_action_antonym = get_action_antonyms(
            annotations=ds.annotation,
            text_key='template',
            additional_antonyms_map=additional_antonyms_map,
            ignored_verbs=ignored_verbs
        )
        print(f'{stage} length after filtering by template action antonyms: {len(annotations_with_action_antonym)}')

        # Field action_antonym_clip_text --> template_action_antonym_clip_text
        for annot in annotations_with_action_antonym:
            annot['template_action_antonym_clip_text'] = annot['action_antonym_clip_text'].capitalize()
            del annot['action_antonym_clip_text']

        # Construct label_action_antonym_clip_text field
        if stage != 'test':
            annotations_with_action_antonym = get_action_antonyms(
                annotations=annotations_with_action_antonym,
                text_key='label',
                additional_antonyms_map=additional_antonyms_map,
                ignored_verbs=ignored_verbs
            )

            for annot in annotations_with_action_antonym: # Field action_antonym_clip_text --> label_action_antonym_clip_text
                annot['label_action_antonym_clip_text'] = annot['action_antonym_clip_text'].capitalize()
                del annot['action_antonym_clip_text']
        else:
            for annot in annotations_with_action_antonym:
                annot['label_action_antonym_clip_text'] = None

        print(f'{stage} length after filtering by label action antonyms: {len(annotations_with_action_antonym)}')

        # Field clip_uid --> id before writing
        for annot in annotations_with_action_antonym:
            annot['id'] = annot['clip_uid']
            del annot['clip_uid']

        # Output file
        os.makedirs(output_ann_path, exist_ok=True)

        if stage != 'test':
            stage = stage if stage != 'val' else 'validation'

            with open(os.path.join(output_ann_path, f'{stage}.json'), 'w') as f:
                json.dump(annotations_with_action_antonym, f)
        else:
            import pandas as pd
            df = pd.DataFrame.from_records(annotations_with_action_antonym)
            df.to_csv(os.path.join(output_ann_path, 'test-answers.csv'), index=False, sep=';')


# %%
if __name__ == "__main__":
    # NOTE: main function for getting action antonyms: get_action_antonyms()
    
    ## usage examples ##    
    random.seed(42)
    logging.basicConfig(level=logging.INFO)
    # Ego4D 
    get_action_antonyms_ego4d(output_ann_path = "<set output path>")
    # SSv2
    get_action_antonyms_ssv2(output_ann_path = "<set output path>")
