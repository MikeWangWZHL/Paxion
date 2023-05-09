import json
import os
import random
from tqdm import tqdm
random.seed(42)  # Set the random seed to ensure reproducibility

### == some helper functions == ###
class PostProcess():
    # handle edge cases and artifacts from the antonym mining
    def __init__(self) -> None:
        additional_antonyms_mapping = json.load(open("additional_antonyms_mapping_ego4d.json"))
        self.post_process_targets = {}
        for value in additional_antonyms_mapping.values():
            if value.endswith("s") or value.endswith("es"):
                self.post_process_targets[value+"es"] = value
        print("post processing target:value:", self.post_process_targets)
    
    def run(self, ann):
        action_antonym_clip_text = ann['action_antonym_clip_text']
        for key,value in self.post_process_targets.items():
            action_antonym_clip_text = action_antonym_clip_text.replace(key,value)
        ann['action_antonym_clip_text'] = action_antonym_clip_text

def filterer(ann):
    if_filter = False
    filtering_verbs = ["keeps","keep","kept"]
    for v in filtering_verbs:
        if v in ann["clip_text"]:
            if_filter = True
            break
    return if_filter 


### == set the input and output paths == ###
processed_annotation_jsonl_name = "<processed_jsonl_name>" # e.g. "egoclip_subset_action_antonyms"
input_jsonl = f"../ego4d/{processed_annotation_jsonl_name}.jsonl"

train_output = f"<output_path>/{processed_annotation_jsonl_name}/train.jsonl"
val_output = f"<output_path>/{processed_annotation_jsonl_name}/val.jsonl"
test_output = f"<output_path>/{processed_annotation_jsonl_name}/test.jsonl"

### == set split ratio == ###
ratios = [0.8,0.1,0.1]


### == run the script == ###
post_processor = PostProcess()

annotations = []
with open(input_jsonl, 'r') as f:
    for line in tqdm(f):
        loaded_ann = json.loads(line)
        post_processor.run(loaded_ann)
        if not filterer(loaded_ann):
            annotations.append(loaded_ann)
        else:
            print("filtered:", loaded_ann['clip_text'])
print(len(annotations))

random.shuffle(annotations)

sizes = [int(len(annotations)*r) for r in ratios]

print(sizes)

train_anns = annotations[:sizes[0]]
val_anns = annotations[sizes[0]:sizes[0]+sizes[1]]
test_anns = annotations[sizes[0]+sizes[1]:]

print("train size:", len(train_anns))
print("val size:", len(val_anns))
print("test size:", len(test_anns))

with open(train_output, 'w') as out:
    for line in train_anns:
        out.write(json.dumps(line))
        out.write("\n")

with open(val_output, 'w') as out:
    for line in val_anns:
        out.write(json.dumps(line))
        out.write("\n")

with open(test_output, 'w') as out:
    for line in test_anns:
        out.write(json.dumps(line))
        out.write("\n")