#!/usr/bin/env python3
#
import sys, os, json
import pdb
import random
import parser

# random.seed(0)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-mode', default ='train_maml')
    # args = parser.parse_args()

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

    # target_dir = data_path
    adapt_dial_num = 400
    # test_dial_num = 200
    # test_dial_num_string = 'other' if test_dial_num > 300 else str(test_dial_num)
    sys.stdout.write('extract ' + str(adapt_dial_num) + ' dialogs for adaptation and other dialogs for test\n')


    for i in range(10):
        random.seed(i)
        target_dir = './data/multi-woz-processed/adapt_' + str(adapt_dial_num) + '/' + str(i) + '/'

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

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


                    random.shuffle(domain_files[domain])

                    # # # extract adaptation data
                    with open(os.path.join(target_dir, 'adapt_data_in_domain_' + domain_name + '.json'), 'w') as minor_file:
                        adapt_data_in_domain = {}
                        for adapt_dial_name in domain_files[domain][:adapt_dial_num]:
                            dial_id = adapt_dial_name.split('.')[0]
                            adapt_data_in_domain[dial_id] = data[dial_id]

                        json.dump(adapt_data_in_domain, minor_file, indent = 4)

                        # sys.stdout.write('complete extracting ' + str(adapt_dial_num) + ' dialogs for adaptation in ' + domain_name + ' domain ...\n')

                    # # # extract all the single-domain data for training
                    with open(os.path.join(target_dir, 'data_in_domain_' + domain_name + '.json'), 'w') as file:
                        data_in_domain = {}

                        for dial_name in domain_files[domain]:
                            dial_id = dial_name.split('.')[0]
                            data_in_domain[dial_id] = data[dial_id]

                        json.dump(data_in_domain, file, indent = 4)

                        # sys.stdout.write('complete extracting ' + str(len(domain_files[domain])) + ' dialogs for training in ' + domain_name + ' domain ...\n\n')

                    # # # extract the rest as test data
                    with open(os.path.join(target_dir, 'test_data_in_domain_' + domain_name + '.json'), 'w') as file:
                        test_data_in_domain = {}

                        # end_idx = min(len(domain_files[domain]), adapt_dial_num + test_dial_num)
                        if len(domain_files[domain]) - adapt_dial_num <= 50:
                            test_data_in_domain = data_in_domain
                            test_dial_num = len(domain_files[domain])
                        else:
                            for test_dial_name in domain_files[domain][adapt_dial_num:]:
                                dial_id = test_dial_name.split('.')[0]
                                test_data_in_domain[dial_id] = data[dial_id]
                            test_dial_num = len(domain_files[domain][adapt_dial_num:])

                        # pdb.set_trace()

                        json.dump(test_data_in_domain, file, indent = 4)

                        # sys.stdout.write('complete extracting ' + str(len(domain_files[domain][adapt_dial_num:])) + ' dialogs for test in ' + domain_name + ' domain ...\n')

                sys.stdout.write('Complete extracting {}/{}/{}'.format(adapt_dial_num,
                                                                       test_dial_num, 
                                                                       len(domain_files[domain])) + ' dialogs for adapt/test/train in ' + domain_name + ' domain ...\n')
        sys.stdout.write('Finish for random set ' + str(i) + ' ...\n\n')


    data_json.close()
    domain_files_json.close()


if __name__ == '__main__':
    main()