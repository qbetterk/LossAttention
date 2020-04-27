#!/usr/bin/env python3
#
import sys, os, json
import pdb


def main():
    data_path = './data/multi-woz-processed/'
    data_file = 'data_for_damd.json'
    data_json = open(os.path.join(data_path, data_file))
    data      = json.loads(data_json.read().lower())
    """
    dict{
        str(dial_id):
        dict{
            'goal':
            dict{
                str(domain): ...
            };
            'log': [
            dict{
                ...
            },
            ...
            ]        
        }
    }
    """


    domain_file_path = 'data/multi-woz-processed/domain_files.json'
    domain_files_json = open(domain_file_path)
    domain_files = json.loads(domain_files_json.read().lower())
    """
    dict{
        str(domain):[
                    str(dial_id) + '.json', 
                    str(dial_id) + '.json', 
                    ...
                    ]
        }
    """
    # pdb.set_trace()

    single_domain_list = []
    for domain in domain_files:
        if domain.endswith('single'):
            domain_name = domain.split('_')[0]
            if domain_name in single_domain_list:
                pdb.set_trace()
            else:
                single_domain_list.append(domain_name)

                # if 'restaurant' in domain:
                #     print(domain)
                #     pdb.set_trace()
                domain_files[domain] = list(set(domain_files[domain]))

                with open(os.path.join(data_path, 'data_in_domain_' + domain_name + '.json'), 'w') as file:
                    data_in_domain = {}
                    for dial_name in domain_files[domain][9:]:
                        dial_id = dial_name.split('.')[0]
                        data_in_domain[dial_id] = data[dial_id]

                    json.dump(data_in_domain, file, indent = 4)

                    sys.stdout.write('complete extracting ' + str(len(domain_files[domain][9:])) + ' dialogs in ' + domain_name + ' domain ...\n')

                with open(os.path.join(data_path, 'minor_data_in_domain_' + domain_name + '.json'), 'w') as minor_file:
                    minor_data_in_domain = {}
                    for minor_dial_name in domain_files[domain][:9]:
                        dial_id = minor_dial_name.split('.')[0]
                        minor_data_in_domain[dial_id] = data[dial_id]

                    json.dump(minor_data_in_domain, minor_file, indent = 4)

                    sys.stdout.write('complete extracting the rest 9 dialogs in ' + domain_name + ' domain ...\n\n')


    data_json.close()
    domain_files_json.close()


if __name__ == '__main__':
    main()