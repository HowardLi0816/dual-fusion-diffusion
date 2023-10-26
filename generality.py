import spacy
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import numpy as np
import json
import os


#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')


'''
def quantify_specificity(sentence):

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    print(doc)
    count = 0
    for ent in doc.ents:
        #print(ent)
        count += len(ent)
    
    numerals = [token for token in doc if token.pos_ == 'NUM']
    # Count of named entities and numerals
    specifics_count = count + len(numerals)
    
    
    max_specifics = len(doc)
    
    specificity_score = (1 - specifics_count / max_specifics) * 10
    if specificity_score < 0:
        specificity_score = 0
    
    return specificity_score
'''

def quantify_specificity(sentence):
    '''
    score between 0(specific) and 10(general)
    '''
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    #print("doc:", doc.ents)
    #count = 0
    #for ent in doc.ents:
    #    print("ent:", ent, len(ent))
    #    count += len(ent)
    #print (count)
    new_count=0
    for token in doc.ents:
        #print (f"token:{token}, {len(token)}")
        for subtoken in token:

            if subtoken.pos_=='NUM':
                continue

            #print (f"subtoken:{subtoken}, {subtoken.pos_}")
            new_count+=1
    #print (new_count)



    numerals = [token for token in doc if token.pos_ == 'NUM']
    # Count of named entities and numerals
    #specifics_count = count + len(numerals)
    specifics_count = new_count + len(numerals)


    max_specifics = len(doc)

    #print (specifics_count / max_specifics)

    specificity_score = (1 - specifics_count / max_specifics) * 10
    if specificity_score < 0:
        specificity_score = 0
    
    return specificity_score



# def quantify_broadness(sentence):
#     words = word_tokenize(sentence)
    
#     # count the total number of hyponyms for each noun
#     total_hyponyms = 0
    
#     for word in words:
#         # Get the synsets for this word
#         synsets = wn.synsets(word, 'n')
#         for synset in synsets:
#             # Get the hyponyms for this synset
#             hyponyms = synset.hyponyms()
#             total_hyponyms += len(hyponyms)
    
#     return total_hyponyms

# def quantify_broadness(sentence):
#     words = word_tokenize(sentence)
#     pos_tags = nltk.pos_tag(words)
#     total_hyponyms = 0
    
#     for word, pos in pos_tags:
#         if pos.startswith('N'):  # If the word is a noun
#             synsets = wn.synsets(word, 'n')  # Get the noun synsets of the word
#             hyponym_count = sum(len(synset.hyponyms()) for synset in synsets)
#             total_hyponyms += hyponym_count
#     print(total_hyponyms)
#     avg_hyponym = total_hyponyms/len(words)
    
#     return avg_hyponym

def quantify_broadness(sentence, all_avg):
    words = word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)

    # Filter out the nouns
    nouns = [word for word, pos in tagged_words if pos.startswith('NN')]

    if not nouns:  # If there are no nouns in the sentence
        return 0

    # Calculate the average hyponyms for the nouns in the sentence
    average_hyponyms_sentence = sum(len(wn.synsets(noun, 'n')[0].hyponyms()) for noun in nouns if wn.synsets(noun, 'n')) / len(words)
    
    broadness = np.interp(average_hyponyms_sentence, (0, all_avg*2), (0, 10))
    return broadness

def get_hyponym_avg():
    """Get an avg count of noun hyponym for each word in WordNet (divided by the total number of words in WordNet)."""
    hyponym_counts = []
    # lemmas = wn.all_lemma_names()
    # total_num = len(set(lemmas))
    #print("Total lemmas in WordNet:", total_num)
    for synset in list(wn.all_synsets('n')):
        
        hyponym_counts.append(len(synset.hyponyms()))
    #avg = np.sum(hyponym_counts)/total_num
    # avg = np.sum(hyponym_counts)/len(list(wn.all_synsets()))
    
    # Assume 30000 words are used in daily life, might change
    avg = np.sum(hyponym_counts)/30000
    return avg

def quantify_tense_modality(sentence):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    
    # Get all verbs
    verbs = [token for token in doc if token.pos_ == 'VERB']
    total_verbs = len(verbs)
    
    # If no verbs are found, we return a neutral score
    if total_verbs == 0:
        return 5.0

    count_present_indicative = 0
    for verb in verbs:
        #print(verb)
        if verb.morph.get('Tense') == ['Pres'] or verb.morph.get('Mood') == ['Ind']:
            #print(verb)
            count_present_indicative += 1
    
    score = 10.0 * (count_present_indicative / total_verbs)

    return score
    
def all_avg_depth():
    #Calculate the average depth of all noun synsets in WordNet
    all_noun_synsets = list(wn.all_synsets('n'))
    average_depth = sum(synset.min_depth() for synset in all_noun_synsets) / len(all_noun_synsets)
    return average_depth
    
def get_abstraction_score(sentence):
    tagged_words = nltk.pos_tag(word_tokenize(sentence))
    
    nouns = [word for word, pos in tagged_words if pos.startswith('NN')]
    
    average_depth = all_avg_depth()
    #print(average_depth)
    depths = []
    for noun in nouns:
        synsets = wn.synsets(noun, 'n')
        if synsets:
            # Get the depth of the first synset (this could be refined)
            depth = synsets[0].min_depth()
            depths.append(depth)
            
    # Calculate the average depth
    average_sentence_depth = sum(depths) / len(depths) if depths else average_depth

    # Calculate the score (in [0, 10]) by comparing the sentence's average depth to the overall average depth
    score = np.interp(average_sentence_depth, (0, average_depth*2), (0, 10))

    return score
    
def quantify_context_dependence(sentence):
    tagged_words = nltk.pos_tag(word_tokenize(sentence))
    total_words = len(tagged_words)
    pronouns = sum(1 for word, pos in tagged_words if pos.startswith('PRP'))
    return np.interp(pronouns, (0, total_words/2), (10, 0))

'''
#sentence = "Versatile and Upgraded Right Angle Depressed Center Wheel Grinder for Various Applications"
sentence = "A Purple and Fuchsia Modern Glam Winery Wedding | Blueberry Phot"
#sentence = "100% Cotton Sateen White Double High thread count 600TC bed Sheet with 2 pillow covers:  Bedroom by FurnishTurf"
#sentence = "It sits on him."
print(quantify_specificity(sentence))
all_avg = get_hyponym_avg()
print(quantify_broadness(sentence, all_avg))
#print(all_avg)
print(quantify_tense_modality(sentence))
print(get_abstraction_score(sentence))
print(quantify_context_dependence(sentence))
'''



rootdir = './laion_10k_data_2/'
#with open(os.path.join(rootdir, "laion_10k_data_2_GPT_5_words_clean_captions.json"),'r') as f:
with open(os.path.join(rootdir, "laion_10k_data_2_GPT_clean_captions.json"),'r') as f:
    json_str = f.read()
    orig = json.loads(json_str)
    
result_dict = {}

for key in orig:
    print(key)
    value = orig[key]
    tmp_list = []
    v = value[0]
    tmp_list.append(v)
    #tmp_list.append(value[1])
    score_dict = {}
    score_dict["specificity_score"] = quantify_specificity(v)
    all_avg = get_hyponym_avg()
    score_dict["broadness_score"] = quantify_broadness(v, all_avg)
    score_dict["tense_modality_score"] = quantify_tense_modality(v)
    score_dict["abstraction_score"] = get_abstraction_score(v)
    #score_dict["context_dependence_score"] = quantify_context_dependence(v)
    
    score_dict["average"] = np.mean(list(score_dict.values()))
            
    tmp_list.append(score_dict)
    
    result_dict[key] = tmp_list

json_object = json.dumps(result_dict, indent=4)

base = rootdir.split('/')[1]
json_path = os.path.join(rootdir, f"{base}_GPT_captions_with5gen_score_update.json")

with open(json_path, "w") as outfile:
    outfile.write(json_object)
    