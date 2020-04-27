#!/usr/bin/env python3
#
import sys, os
import json
import pdb

domains = [
            'attraction',
            'train',
            'taxi',
            'restaurant',
            'hospital',
            'hotel',
            'police'
            ]

for domain in domains:
    rewrite_file_path = './data/multi-woz-processed/data_in_domain_' + domain + '_rewrite.json'
    minor_data_path = './data/multi-woz-processed/minor_data_in_domain_' + domain + '.json'
    rewrite_minor_data_path = './data/multi-woz-processed/minor_data_in_domain_' + domain + '_rewrite.json'

    with open(rewrite_file_path) as rf:
        all_dials = json.loads(rf.read().lower())
    with open(minor_data_path) as md :
        minor_data = json.loads(md.read().lower())


    # # # split out adapt data(minor)
    rewrite_minor_data = {}
    for dial_id in minor_data:
        rewrite_minor_data[dial_id] = all_dials[dial_id]

    with open(rewrite_minor_data_path, 'w+') as rmd:
        json.dump(rewrite_minor_data, rmd, indent = 2)

    # # # shrink train data
    for dial_id in minor_data:
        del all_dials[dial_id]

    with open(rewrite_file_path, 'w+') as rf:
        json.dump(all_dials, rf, indent = 2)

