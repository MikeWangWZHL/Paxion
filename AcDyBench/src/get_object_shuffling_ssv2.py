import json
import os
from collections import defaultdict    
from nltk.corpus import wordnet
import nltk
import spacy
import pyinflect
from tqdm import tqdm
import random

def get_object_shuffling_ssv2(output_path = None):
    ann_path = "../ssv2/antonyms/validation.json"
    assert output_path is not None
    object_taxonomy = json.load(open("../ssv2/object_taxonomy.json"))
    print(len(object_taxonomy))

    val_annotations = json.load(open(ann_path))

    output_annotations = []
    for item in tqdm(val_annotations):
        orig_objects = item['placeholders']
        
        cand_object_taxonomy = object_taxonomy.copy()
        for orig in orig_objects:
            if orig in cand_object_taxonomy:
                cand_object_taxonomy.remove(orig)
        
        cand_objects = random.sample(cand_object_taxonomy, len(orig_objects))
        
        object_shuffled_text = item['label'].lower()
        for i, orig in enumerate(orig_objects):
            orig = orig.lower()
            # Find the index of the first occurrence of the substring
            index = object_shuffled_text.find(orig)
            assert index != -1
            # Replace the first occurrence of the substring with a new string
            object_shuffled_text = object_shuffled_text[:index] + cand_objects[i] + object_shuffled_text[index+len(orig):]
        
        assert object_shuffled_text != item['label']
        item['label_object_shuffled_clip_text'] = object_shuffled_text
        
        output_annotations.append(item)

    assert len(output_annotations) == len(val_annotations)
    with open(output_path, 'w') as o:
        json.dump(output_annotations, o, indent=4)



if __name__ == "__main__":

    random.seed(42)
    get_object_shuffling_ssv2(output_path = "<output path for processed validation set>")