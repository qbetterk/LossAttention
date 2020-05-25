#!/usr/bin/env python3
#
import sys, os, csv
import pdb
import argparse


def extract_from_dir(dir_path):
    results = []
    # results = {'slot_acc':[], 'slot_f1':[], 'act_f1':[], 'inform':[], 'success':[], 'bleu':[]}
    for test_dir in sorted(os.listdir(dir_path)):
        if '9test_s' in test_dir and os.path.isdir(os.path.join(dir_path, test_dir)):
            for file in os.listdir(os.path.join(dir_path, test_dir)):
                if 'log_adapt_' in file:
                    file_path = os.path.join(dir_path, test_dir, file)
                    results.append({})
                    for line in open(file_path).readlines():
                        if '[DST] joint goal' in line:
                            split_line = line.split()
                            # if len(split_line) < 6:
                            #     pdb.set_trace()
                            results[-1]['slot_acc'] = float(split_line[5])
                            results[-1]['slot_f1'] = float(split_line[8])
                            results[-1]['act_f1'] = float(split_line[-1])

                        if '[CTR] match' in line:
                            split_line = line.split()
                            results[-1]['inform'] = float(split_line[2])
                            results[-1]['success'] = float(split_line[4])
                            results[-1]['bleu'] = float(split_line[6])

                    if results[-1] == {}:
                        results.remove({})
                    else:
                        results[-1]['domain'] = test_dir
                # if results == []:
                #     pdb.set_trace()
    return results


def extract_from_file(file_path):
    pass


def average(results):
    avg_result = {}
    for key in results[0]:
        if key != 'domain':
            # # # pdb.set_trace()
            # tmp_list = []
            # for result in results:
            #     if key not in result:
            #         pdb.set_trace()
            #     tmp_list.append(result[key])

            tmp_list = [result[key] for result in results]
            avg_result[key] = float(sum(tmp_list)) / len(tmp_list)
            # avg_result[key] = float(sum(sorted(tmp_list)[1:-1])) / (len(tmp_list) - 2)
    if 'domain' in results[0]:
        avg_result['domain'] = 'avg'
    return avg_result

def write_to_csv(output_path, results):
    with open(output_path, 'w') as rf:
        writer = csv.DictWriter(rf, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        writer.writerows([average(results)])

def write_one_exp(dir_path):
    results = extract_from_dir(dir_path)

    output_path = os.path.join(dir_path, 'combined_result_9.csv')
    write_to_csv(output_path, results)




def write_multi_dom(dir_path):

    output_path = os.path.join(dir_path, 'combined_result.csv')
    with open(output_path, 'w') as rf:
        fields = ['domain', 'slot_acc', 'slot_f1', 'act_f1', 'inform', 'success', 'bleu']
        writer = csv.DictWriter(rf, fieldnames=fields)
        writer.writeheader()

        for exp in sorted(os.listdir(dir_path)):
            domain = exp.split('_')[1]

            exp_path = os.path.join(dir_path, exp)
            if not os.path.isdir(exp_path):
                continue

            results = extract_from_dir(exp_path)

            if results == []:
                continue

            writer.writerows([{'domain':exp}])
            writer.writerows(results)
            writer.writerows([average(results)])

            writer.writerows([{}])
            writer.writerows([{}])






def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-multi', default=0)
    parser.add_argument('-path', default='')
    args = parser.parse_args()

    # path = 'experiments/w-1_restaurant_sd333_lr0.005_mlr0.005_bs32_vs3k_tmp5'
    # path = 'experiments/te0_td0_w-1_restaurant_sd333_lr0.005_mlr0.005_bs32_vs3k_tenl2_tnh5_from_interaction'
    # path = os.path.join(args.path, 'w0_restaurant_sd333_lr0.005_mlr0.005_bs32_vs3k_baseline/')
    # path = './experiments/w-1_hotel_sd333_lr0.005_mlr0.005_bs32_vs3k_tenl2_tnh5/'
    # path = './experiments/te0_td0_w-1_train_sd333_lr0.005_mlr0.005_bs32_vs3k_from_interaction/'
    path = args.path

    # pdb.set_trace()
    if not int(args.multi):
        write_one_exp(path)
    else:
        write_multi_dom(path)

if __name__ == "__main__":
    main()