from collections import Counter, defaultdict
import glob
import re
import spacy
import string
import numpy as np
import pandas as pd
import gender_guesser.detector as gender

d = gender.Detector(case_sensitive=False)
nlp = spacy.load("en_core_web_lg")
label_dict = {'male': 'male', 'mostly_male': 'male', 'female': 'female', 'mostly_female': 'female',
              'both': 'both'}
              
def split_string(text):
    # Remove leading and trailing punctuation
    text = text.strip(string.punctuation)
    
    # Replace punctuation with spaces
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    
    # Split the string into a list of words
    words = text.split()
    
    return words


def label_gender(x):
    return [d.get_gender(i) for i in split_string(str(x)) if d.get_gender(i) != 'unknown']


def detect_people(text):
    doc = nlp(text)
    people = list([ent.text for ent in doc.ents if ent.label_ == 'PERSON'])
    
    genders = []
    for x in people:
        if len(label_gender(x)) >= 1:
            genders.append(label_gender(x)[0])
    
    return people, genders


def detect_gender_words(string):
    male_gender_words = ["he", "hes", "he's", "him", "his", "himself", "male", "males", 
                     "man", "mans", "men", "mens", "boy", "boys", "gent", "gents", 
                     "gentleman", "gentlemans", "gentlemen", "gentlemens"]
    female_gender_words = ["she", "shes", "she's", "her", "hers", "her's", "herself", 
                       "female", "females", "woman", "womans", "women", "womens", 
                       "girl", "girls", "lady", "ladies"]
    
    string = string.lower()
    # Build a regular expression pattern that matches any of the gender_words
    pattern = r"\b(" + "|".join(male_gender_words + female_gender_words) + r")\b"
    
    # Use the re module to find all matches of the pattern in the string
    matches = re.findall(pattern, string)

    # Determine the gender of the gender_words found
    gender = "none"
    if any(match in male_gender_words for match in matches) and not any(match in female_gender_words for match in matches):
        gender = "male"
    elif any(match in female_gender_words for match in matches) and not any(match in male_gender_words for match in matches):
        gender = "female"
    elif any(match in female_gender_words for match in matches) and any(match in male_gender_words for match in matches):
        gender = 'both'
    return gender


def combine_labels(people, gender_words):
    final = people + [gender_words]
    final = set([label_dict[x] for x in final if x != 'none' and x != 'andy'])
    if final == set(['male']) or final == set(['female']):
        label = list(final)[0]
    # both male and female gender indicators are present
    elif 'both' in final or ('male' in final and 'female' in final): 
        label = 'both'
    # neither male or female gender indicators are present
    else: 
        label = 'unknown'
    return final, label