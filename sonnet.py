import os
import re
import random
import numpy as np
from HMM_numpy import unsupervised_HMM
from Utility import Utility
from HMM_helper import obs_map_reverser

def parse_observations(text):
    # Convert text to dataset.
    lines = [re.findall(r"[\w\-']+|[.,!?;:]", line) for line in text.split('\n') if line.split()]

    obs_counter = 0
    obs = []
    obs_map = {}

    for line in lines:
        obs_elem = []

        for i in range(len(line)):
            word = line[i]
            if i == 0:
                word = word.lower()
            if word not in obs_map:
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1

            # Add the encoded word.
            obs_elem.append(obs_map[word])

        # Add the encoded sequence.
        obs.append(obs_elem)

    return obs, obs_map

def create_pron_map(text, obs_map):
    pron_map = {}
    lines = [line.split() for line in text.split('\n') if line.split()]
    for line in lines:
        if line[0].lower() in obs_map:
            word = line[0].lower()
        elif line[0].lower().capitalize() in obs_map:
            word = line[0].lower().capitalize()
        else:
            continue
        pron_map[obs_map[word]] = line[1:]

    return pron_map

def create_syl_map(text, obs_map):
    syl_map = {}
    lines = [line.split() for line in text.split('\n') if line.split()]
    rand_syl = random.randint(1, 2)
    for line in lines:
        if line[0] in obs_map:
            word = line[0]
        elif line[0].capitalize() in obs_map:
            word = line[0].capitalize()
        else:
            continue
        if len(line) == 2:
            syl_map[obs_map[word]] = [-1, int(line[1])]
        elif len(line) == 3:
            if 'E' in line[1]:
                syl_map[obs_map[word]] = [int(line[1][1]), int(line[2])]
            elif 'E' in line[2]:
                syl_map[obs_map[word]] = [int(line[1]), int(line[2][1])]
            else:
                syl_map[obs_map[word]] = [-1, int(line[rand_syl])]
        else:
            print("Error in parsing syllable dictionary, line {}".format(line))
    return syl_map

def sample_sentence(hmm, obs_map, syl_map, pron_map, n_lines=14):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    emission, states = hmm.generate_emission(n_lines, syl_map)
    poem = []
    for i in range(n_lines):
        line = [obs_map_r[j] for j in emission[i]]
        poem.append(' '.join(line).capitalize())

    return '\n'.join(poem)

if __name__ == '__main__':
    # Parse the sonnets
    text = open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read()
    obs, obs_map = parse_observations(text)
    # Create syllable count dictionary
    text2 = open(os.path.join(os.getcwd(), 'Syllable_dictionary.txt')).read()
    syl_map = create_syl_map(text2, obs_map)
    # Create pronunciation dictionary
    text3 = open(os.path.join(os.getcwd(), 'cmudict-0.txt')).read()
    pron_map = create_pron_map(text3, obs_map)
    for key in obs_map:
        if obs_map[key] not in pron_map:
            print(key)
    # Train unsupervised HMM
    hmm = unsupervised_HMM(obs, 1, 1)
    print(sample_sentence(hmm, obs_map, syl_map, pron_map, n_lines=14))