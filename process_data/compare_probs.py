#!/usr/bin/env python3
#
import sys, os
import json
import pdb
from tqdm import tqdm

data_dir = './data/multi-woz-processed/'
target_dir = './data/multi-woz-processed/compare/'
domains = [
            'attraction',
            'train',
            'taxi',
            'restaurant',
            'hospital',
            'hotel',
            'police'
            ]
# domains = ['restaurant']

for domain in tqdm(domains):
    rewrite_file_path = data_dir + 'data_in_domain_' + domain + '_rewrite.json'
    compare_rewrite_file_path = target_dir + 'data_in_domain_' + domain + '_rewrite_compare.json'

    with open(rewrite_file_path) as rf:
        domain_dials = json.loads(rf.read().lower())


    shrink_dials = {}
    for dial_id in domain_dials:
        shrink_dials[dial_id] = {'log':[]}
        for turn_num in range(len(domain_dials[dial_id]['log'])):
            shrink_dials[dial_id]['log'].append({})

            # shrink_dials[dial_id]['log'][turn_num]['user'] = domain_dials[dial_id]['log'][turn_num]['user']
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

            # mixed_probs_resp = domain_dials[dial_id]['log'][turn_num]['mixed_probs_resp_2']
            # shrink_dials[dial_id]['log'][turn_num]['mix2'] = '  '.join(['{:.2f}'.format(prob) for prob in mixed_probs_resp])

            # mixed_probs_resp = domain_dials[dial_id]['log'][turn_num]['mixed_probs_resp_3']
            # shrink_dials[dial_id]['log'][turn_num]['mix3'] = '  '.join(['{:.2f}'.format(prob) for prob in mixed_probs_resp])

            # mixed_probs_resp = domain_dials[dial_id]['log'][turn_num]['mixed_probs_resp_4']
            # shrink_dials[dial_id]['log'][turn_num]['mix4'] = '  '.join(['{:.2f}'.format(prob) for prob in mixed_probs_resp])

            # mixed_probs_resp = domain_dials[dial_id]['log'][turn_num]['mixed_probs_resp_5']
            # shrink_dials[dial_id]['log'][turn_num]['mix5'] = '  '.join(['{:.2f}'.format(prob) for prob in mixed_probs_resp])
                


    with open(compare_rewrite_file_path, 'w+') as crf:
        json.dump(shrink_dials, crf, indent = 2)