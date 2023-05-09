# Code Description
You need to install `spacy`, `nltk` and `pyinflect` for using these script.

## get action antonyms
- `get_action_antonyms_ego4d_ssv2.py`: contains example code for getting altered text sentences with verbs replaced with their antonyms.
    -  `get_action_antonyms()` is the main function for finding action antonyms given an original text annotation;
    -  `get_action_antonyms_ego4d()` shows an example and some comments on processing Ego4d annotations;
        -  variable `ADDITIONAL_ANTONYYMS_MAPPING` should contain a table of semi-automatically constructed verb-antonym pairs which can be dataset specific; (For Ego4d, we first get a list of verbs from the provided taxonomy and then ask ChatGPT to generate antonym for each of them, and then manually clean up.)
        -  variable `IGNORED_VERBS` should contain a list of verbs that does not have a good antonym, this is manually created;
    -   `get_action_antonyms_ssv2()` shows a similar example on ssv2

## get shuffled objects
- `get_object_shuffling_*`: contains example code for getting altered text sentences with object names replaced by a random object based on dataset taxonomy (Ego4d and SSv2)