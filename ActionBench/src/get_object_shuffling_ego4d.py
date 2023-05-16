import json
import os
from collections import defaultdict    
from nltk.corpus import wordnet
import nltk
import spacy
import pyinflect
from tqdm import tqdm
import random


def load_annotation_ego4d(dataset_name, vis_root, ann_path):
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

def get_object_shuffling_ego4d(output_root = None):
    assert output_root is not None
    os.makedirs(output_root, exist_ok=True)

    vis_root = "../../datasets/Ego4D/video_clips/clips_downsampled_5fps_downsized_224x224"
    
    for split in ['val','test']:
        
        ann_path = f"../ego4d/egoclip_subset_action_antonyms_train_val_test_split/{split}.jsonl"
        
        # load annotation
        annotation = load_annotation_ego4d("egoclip_subset", vis_root, ann_path)
        print("loaded ann:",ann_path)
        print("size:",len(annotation))

        # init spacy nlp pipeline
        # nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        nlp = spacy.load("en_core_web_trf", disable=["ner", "textcat"]) # this one is slower but with higher accurarcy

        # load taxonomy
        label_2_group, noun_2_label = load_noun_taxonomy_antonyms()

        annotations_with_object_shuffling = []
        # filter out ann that has at least one action
        # for idx, sample in enumerate(annotation):
        for sample in tqdm(annotation):
            # check if at least one verb and one object exists    
            assert 'tag_noun' in sample and 'tag_verb' in sample
            noun_idx = eval(sample['tag_noun']) # specific for egoclip annotation
            verb_idx = eval(sample['tag_verb']) # specific for egoclip annotation
            
            # for other datasets, you may need to do this filtering after processing with spacy
            assert len(verb_idx)>=1 and len(noun_idx)>=1
            
            # load annotation text
            original_clip_text = sample["clip_text"]

            # process with spacy
            doc = nlp(original_clip_text)

            object_shuffled_clip_text = original_clip_text
            for token in doc:
                # try to replace the object that is interacted with an action
                if token.pos_ == "NOUN" and token.head.pos_ == "VERB" and token.dep_ == 'dobj':
                    object_cand = token.text

                    # try find it in the manual mapping table
                    key = object_cand.lower()
                    if key in noun_2_label:
                        this_group_label = noun_2_label
                        all_other_labels = [l for l in label_2_group.keys() if l != this_group_label]
                        # randomly select an object from another group
                        rand_group_label = random.choice(all_other_labels)
                        object_cand = random.choice(label_2_group[rand_group_label])
                    else:
                        all_group_labels = list(label_2_group.keys())
                        # randomly select an object from another group
                        rand_group_label = random.choice(all_group_labels)
                        object_cand = random.choice(label_2_group[rand_group_label])

                    if object_cand != token.text:
                        object_shuffled_clip_text = object_shuffled_clip_text.replace(token.text, object_cand)
                        # print(f"{token.text} -> {object_cand}")

            # if we successfully found and replaced verb with its antonym  
            if original_clip_text != object_shuffled_clip_text:
                sample["object_shuffled_clip_text"] = object_shuffled_clip_text
                annotations_with_object_shuffling.append(sample)
            # else:
            #     print("Not able to shuffle object:", original_clip_text)
            # break

        print("output ann split:", split)
        print("output ann size:", len(annotations_with_object_shuffling))
        print()
        # output file
        output_ann_path = f"{output_root}/{split}.jsonl"
        with open(output_ann_path, 'w') as o:
            for line in annotations_with_object_shuffling:
                o.write(json.dumps(line))
                o.write('\n')

def load_noun_taxonomy_antonyms():
    # check if the verbs in the taxonomy all can find some antonyms, if not, we manually assign to them and add in the ADDITIONAL_ANTONYYMS_MAPPING
    import pandas as pd
    input_taxonomy_csv = "../ego4d/narration_noun_taxonomy.csv"
    df = pd.read_csv(input_taxonomy_csv)
    noun_2_label = {}
    label_2_group = {}
    for index, row in df.iterrows():
        group = eval(row['group'])
        label = row['label']
        label_2_group[label] = [t.replace("_"," ").lower() for t in group]
        for noun in label_2_group[label]:
            noun_2_label[noun] = label
    return label_2_group, noun_2_label


if __name__ == "__main__":

    random.seed(42)
    get_object_shuffling_ego4d(output_root="<set up output dir>")