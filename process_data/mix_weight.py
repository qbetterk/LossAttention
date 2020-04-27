#!/usr/bin/env python3
#
import sys, os
import json
import pdb
from tqdm import tqdm

source_data_dir = '../token_reweight/data/multi-woz-processed/'
target_data_dir = './data/multi-woz-processed/'
domains = [
            'attraction',
            'train',
            'taxi',
            'restaurant',
            'hospital',
            'hotel',
            'police'
            ]

for domain in tqdm(domains):

    # # # mix probs from different lm together
    source_rewrite_file_path = source_data_dir + 'data_in_domain_' + domain + '_rewrite.json'
    target_rewrite_file_path = target_data_dir + 'data_in_domain_' + domain + '_rewrite.json'

    with open(source_rewrite_file_path) as rf:
        domain_dials = json.loads(rf.read().lower())

    for dial_id in domain_dials:
        for turn_num in range(len(domain_dials[dial_id]['log'])):
            sum_probs = []
            sum_probs_inv = []
            for prob_dom in domains:
                if prob_dom != domain:
                    probs = domain_dials[dial_id]['log'][turn_num][prob_dom + '_lm_probs_resp']
                    probs = probs
                    probs_inv = [0.1 / (0.1 + i) for i in probs]
                    if sum_probs == []:
                        sum_probs = probs
                        sum_probs_inv = probs_inv
                    else:
                        sum_probs = [sum(x) for x in zip(sum_probs, probs)]
                        sum_probs_inv = [sum(x) for x in zip(sum_probs_inv, probs_inv)]
            self_probs = domain_dials[dial_id]['log'][turn_num][domain + '_lm_probs_resp']
            self_probs_inv = [0.1 / (0.1 + i) for i in self_probs]

            mixed_probs_resp_1 = [x / 6 for x in sum_probs]
            mixed_probs_resp_2 = [x / 6 for x in sum_probs_inv]
            mixed_probs_resp_3 = [sum(x) / 7 for x in zip(sum_probs, self_probs)]
            mixed_probs_resp_4 = [x[0]/x[1] for x in zip(mixed_probs_resp_1, self_probs)]
            mixed_probs_resp_5 = [x[0]/(x[0]+x[1]) for x in zip(mixed_probs_resp_1, self_probs)]

            domain_dials[dial_id]['log'][turn_num]['mixed_probs_resp_1'] = mixed_probs_resp_1
            domain_dials[dial_id]['log'][turn_num]['mixed_probs_resp_2'] = mixed_probs_resp_2
            domain_dials[dial_id]['log'][turn_num]['mixed_probs_resp_3'] = mixed_probs_resp_3
            domain_dials[dial_id]['log'][turn_num]['mixed_probs_resp_4'] = mixed_probs_resp_4
            domain_dials[dial_id]['log'][turn_num]['mixed_probs_resp_5'] = mixed_probs_resp_5
                

    with open(target_rewrite_file_path, 'w+') as rf:
        json.dump(domain_dials, rf, indent = 2)




    # # # split data into train and adjust set
    rewrite_major_file_path = target_data_dir + 'major_data_in_domain_' + domain + '_rewrite.json'
    minor_data_path         = target_data_dir + 'minor_data_in_domain_' + domain + '.json'
    rewrite_minor_data_path = target_data_dir + 'minor_data_in_domain_' + domain + '_rewrite.json'

    with open(target_rewrite_file_path) as rf:
        domain_dials = json.loads(rf.read().lower())
    with open(minor_data_path) as md :
        minor_data = json.loads(md.read().lower())


    # # # split out adapt data(minor)
    rewrite_minor_data = {}
    for dial_id in minor_data:
        rewrite_minor_data[dial_id] = domain_dials[dial_id]

    with open(rewrite_minor_data_path, 'w+') as rmd:
        json.dump(rewrite_minor_data, rmd, indent = 2)

    # # # shrink train data
    for dial_id in minor_data:
        del domain_dials[dial_id]

    with open(rewrite_major_file_path, 'w+') as rf:
        json.dump(domain_dials, rf, indent = 2)



# # # generate the shrink data for compare scores
data_dir = './data/multi-woz-processed/'
compare_dir = './data/multi-woz-processed/compare/'
domains = [
            'attraction',
            'train',
            'taxi',
            'restaurant',
            'hospital',
            'hotel',
            'police'
            ]

for domain in ['restaurant']:#tqdm(domains):
    rewrite_file_path = data_dir + 'data_in_domain_' + domain + '_rewrite.json'
    compare_rewrite_file_path = compare_dir + 'data_in_domain_' + domain + '_rewrite_compare.json'

    with open(rewrite_file_path) as rf:
        domain_dials = json.loads(rf.read().lower())


    shrink_dials = {}
    for dial_id in domain_dials:
        shrink_dials[dial_id] = {'log':[]}
        for turn_num in range(len(domain_dials[dial_id]['log'])):
            shrink_dials[dial_id]['log'].append({})

            shrink_dials[dial_id]['log'][turn_num]['user'] = domain_dials[dial_id]['log'][turn_num]['user']
            shrink_dials[dial_id]['log'][turn_num]['user_delex'] = domain_dials[dial_id]['log'][turn_num]['user_delex']
            # shrink_dials[dial_id]['log'][turn_num]['sys_act'] = domain_dials[dial_id]['log'][turn_num]['sys_act']
            shrink_dials[dial_id]['log'][turn_num]['resp'] = domain_dials[dial_id]['log'][turn_num]['resp']


            for prob_dom in domains:
                if prob_dom == domain:
                    probs_resp_dom = domain_dials[dial_id]['log'][turn_num][prob_dom + '_lm_probs_resp']
                    shrink_dials[dial_id]['log'][turn_num][prob_dom[:4]] = \
                        '  '.join(['{:.2f}'.format(prob) for prob in probs_resp_dom])

            for prob_dom in domains:
                if prob_dom != domain:
                    probs_resp_dom = domain_dials[dial_id]['log'][turn_num][prob_dom + '_lm_probs_resp']
                    shrink_dials[dial_id]['log'][turn_num][prob_dom[:4]] = \
                        '  '.join(['{:.2f}'.format(prob) for prob in probs_resp_dom])
            
            shrink_dials[dial_id]['log'][turn_num]['----'] = '  '

            for i in range(10):
                if 'mixed_probs_resp_'+str(i) in domain_dials[dial_id]['log'][turn_num]:
                    mixed_probs_resp = domain_dials[dial_id]['log'][turn_num]['mixed_probs_resp_'+str(i)]
                    shrink_dials[dial_id]['log'][turn_num]['mix'+str(i)] = '  '.join(['{:.2f}'.format(prob) for prob in mixed_probs_resp])


    with open(compare_rewrite_file_path, 'w+') as crf:
        json.dump(shrink_dials, crf, indent = 2)
