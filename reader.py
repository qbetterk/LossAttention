import numpy as np
import os, csv, random, logging, json
import spacy
import utils, ontology
from copy import deepcopy
from collections import OrderedDict
from db_ops import MultiWozDB
from config import global_config as cfg

import pdb


random.seed(0)
np.random.seed(0)

class _ReaderBase(object):

    def __init__(self):
        self.train, self.dev, self.test, self.adapt = [], [], [], []
        self.vocab = None
        self.db = None

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5: del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i:i[0]))

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == cfg.batch_size:
                # print('batch size: %d, batch num +1'%(len(batch)))
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        # print('last batch size: %d, batch num +1'%(len(batch)))
        if (len(batch)%len(cfg.cuda_device)) != 0:
            batch = batch[:-(len(batch)%len(cfg.cuda_device))]
        if len(batch) > 0.5 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def inverse_transpose_batch(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = {}
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialogs[dial_id] = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if key == 'dial_id':
                        continue
                    value = v_list[idx_in_batch]
                    if key == 'pointer' and self.db is not None:
                        turn_domain = turn_batch['turn_domain'][idx_in_batch][-1]
                        value = self.db.pointerBack(value, turn_domain)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs

    def get_batches(self, set_name):
        global dia_count
        # log_str = ''
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        turn_bucket = self._bucket_by_turn(dial)
        all_batches = []
        for k in turn_bucket:
            if set_name != 'test' and k==1 or k>=17:
                continue
            batches = self._construct_mini_batch(turn_bucket[k])
            # log_str += "turn num:%d, dial num: %d, batch num: %d last batch len: %d\n"%(
                    # k, len(turn_bucket[k]), len(batches), len(batches[-1]))
            all_batches += batches
        # log_str += 'total batch num: %d\n'%len(all_batches)
        random.shuffle(all_batches)
        for i, batch in enumerate(all_batches):
            yield self.transpose_batch(batch)

    def _construct_mini_batch_turn(self, turns):
        all_batches = []
        batch = []
        for turn in turns:
            batch.append(turn)
            if len(batch) == cfg.batch_size:
                all_batches.append(batch)
                batch = []
        # if size of remainder < 1/2 batch_size, 
        # just put them in the previous batch, 
        # otherwise form a new batch
        if len(batch) > 0.5 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches) != 0:
            all_batches[-1].extend(batch)
        return all_batches

    def _transpose_batch_turn(self, batch):
        turn_trans = {}
        for turn in batch:
            for key in turn.keys():
                if key not in turn_trans:
                    turn_trans[key] = []
                turn_trans[key].append(turn[key])
        return turn_trans

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == cfg.batch_size:
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        if len(batch) > 0.5 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def _transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def _mark_batch_as_supervised(self, all_batches):
        supervised_num = int(len(all_batches) * cfg.spv_proportion / 100)
        for i, batch in enumerate(all_batches):
            for dial in batch:
                for turn in dial:
                    turn['supervised'] = i < supervised_num
                    if not turn['supervised']:
                        turn['degree'] = [0.] * cfg.degree_size  # unsupervised learning. DB degree should be unknown
        return all_batches

    def _transpose_batch_maml(self, batch, dial_len):
        """
        batch = {dom1:[dial[turn1, turn2, ...], dial2, ...]; ...}
        """
        dial_batch = []
        for turn_num in range(dial_len):
            turn_batch = {}
            for dom in batch:
                if dom not in turn_batch:
                    turn_batch[dom] = {}
                for dial in batch[dom]:
                    turn = dial[turn_num]
                    for key in turn:
                        if key not in turn_batch[dom]:
                            turn_batch[dom][key] = []
                        turn_batch[dom][key].append(turn[key])
            dial_batch.append(turn_batch)
        return dial_batch

    def _construct_mini_batch_maml(self, dials):
        dials_in_batch = []
        for length in dials:
            if len(dials[length]) <= 2:
                continue

            batch_num = max([len(dials[length][dom]) for dom in dials[length]])
            dial_batch = {}
            for idx in range(batch_num):
                
                for dom in dials[length]:
                    if dom not in dial_batch:
                        dial_batch[dom] = []
                    if idx  < len(dials[length][dom]):
                        dial = dials[length][dom][idx]
                    else: # random pick on dialog if num of dial in this dom is limited
                        dial = random.choice(dials[length][dom])
                    dial_batch[dom].append(dial)

                if (idx+1)%cfg.batch_size == 0:
                    dials_in_batch.append(self._transpose_batch_maml(dial_batch, length))
                    dial_batch = {}
            if dial_batch != {}:
                dials_in_batch.append(self._transpose_batch_maml(dial_batch, length))

        return dials_in_batch

    def _cluster_dial_by_length_maml(self, total_dial_domain):
        dial_clustered = {}
        for domain_idx in range(len(total_dial_domain)):
            for dial in total_dial_domain[domain_idx]:
                dial_length = len(dial)
                if dial_length not in dial_clustered:
                    dial_clustered[dial_length] = {}
                if domain_idx not in dial_clustered[dial_length]:
                    dial_clustered[dial_length][domain_idx] = []
                dial_clustered[dial_length][domain_idx].append(dial)

        for k in dial_clustered:
            logging.debug("dialog length " + \
                            str(k) +         \
                          " with instances: " + \
                          " ".join([str(len(dial_clustered[k][i])) for i in dial_clustered[k]]))

        return dial_clustered

    def mini_batch_iterator_maml_supervised(self, set_name):

        """
        self.train = [[[{}(turn), ...](dial), ...](domain), ...]

        target:  [dial[]]
        """

        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev, 'adapt': self.adapt}
        total_dial_domain = name_to_set[set_name]


        if set_name == 'train':
            # # cluster dial by dial_length
            dials_clustered = self._cluster_dial_by_length_maml(total_dial_domain)

            dials_in_batch = self._construct_mini_batch_maml(dials_clustered)

            for dial_batch in dials_in_batch:
                yield dial_batch

        elif set_name == 'dev':
            dials = []
            for domain in range(len(total_dial_domain)):
                dials += total_dial_domain[domain]

            turn_bucket = self._bucket_by_turn(dials)
            all_batches = []
            for k in turn_bucket:
                batches = self._construct_mini_batch(turn_bucket[k])
                all_batches += batches
            self._mark_batch_as_supervised(all_batches)
            random.shuffle(all_batches)
            for i, batch in enumerate(all_batches):
                yield self._transpose_batch(batch)

        else:
            dials = total_dial_domain

            turn_bucket = self._bucket_by_turn(dials)
            all_batches = []
            for k in turn_bucket:
                batches = self._construct_mini_batch(turn_bucket[k])
                all_batches += batches
            self._mark_batch_as_supervised(all_batches)
            random.shuffle(all_batches)
            for i, batch in enumerate(all_batches):
                yield self._transpose_batch(batch)

    def save_result(self, write_mode, results, field, write_title=False):
        with open(cfg.result_path, write_mode) as rf:
            if write_title:
                rf.write(write_title+'\n')
            writer = csv.DictWriter(rf, fieldnames=field)
            writer.writeheader()
            writer.writerows(results)
        return None

    def save_result_report(self, results):
        ctr_save_path = cfg.result_path[:-4] + '_report_ctr%s.csv'%cfg.seed
        write_title = False if os.path.exists(ctr_save_path) else True
        if cfg.aspn_decode_mode == 'greedy':
            setting = ''
        elif cfg.aspn_decode_mode == 'beam':
            setting = 'width=%s'%str(cfg.beam_width)
            if cfg.beam_diverse_param>0:
                setting += ', penalty=%s'%str(cfg.beam_diverse_param)
        elif cfg.aspn_decode_mode == 'topk_sampling':
            setting = 'topk=%s'%str(cfg.topk_num)
        elif cfg.aspn_decode_mode == 'nucleur_sampling':
            setting = 'p=%s'%str(cfg.nucleur_p)
        res = {'exp': cfg.eval_load_path,
               'true_bspn':cfg.use_true_curr_bspn,
               'true_aspn': cfg.use_true_curr_aspn,
               'decode': cfg.aspn_decode_mode,
               'param':setting,
               'nbest': cfg.nbest,
               'selection_sheme': cfg.act_selection_scheme,
               'match': results[0]['match'],
               'success': results[0]['success'],
               'bleu': results[0]['bleu'],
               'act_f1': results[0]['act_f1'],
               'avg_act_num': results[0]['avg_act_num'],
               'avg_diverse': results[0]['avg_diverse_score'],
               'slot_acc': results[0]['slot_acc'],
               'slot_f1': results[0]['slot_f1']}
        with open(ctr_save_path, 'a') as rf:
            writer = csv.DictWriter(rf, fieldnames=list(res.keys()))
            if write_title:
                writer.writeheader()
            writer.writerows([res])


class MultiWozReader(_ReaderBase):
    def __init__(self, maml=False):
        super().__init__()
        self.nlp = spacy.load('en_core_web_sm')
        self.db = MultiWozDB(cfg.dbs)
        self.vocab_size = self._build_vocab()
        self.domain_files = json.loads(open(cfg.domain_file_path, 'r').read())
        self.slot_value_set = json.loads(open(cfg.slot_value_set_path, 'r').read())
        if cfg.multi_acts_training:
            self.multi_acts = json.loads(open(cfg.multi_acts_path, 'r').read())

        test_list = [l.strip().lower() for l in open(cfg.test_list, 'r').readlines()]
        dev_list = [l.strip().lower() for l in open(cfg.dev_list, 'r').readlines()]
        self.dev_files, self.test_files = {}, {}
        for fn in test_list:
            self.test_files[fn.replace('.json', '')] = 1
        for fn in dev_list:
            self.dev_files[fn.replace('.json', '')] = 1

        self.exp_files = {}
        if 'all' not in cfg.exp_domains:
            for domain in cfg.exp_domains:
                fn_list = self.domain_files.get(domain)
                if not fn_list:
                    raise ValueError('[%s] is an invalid experiment setting'%domain)
                for fn in fn_list:
                    self.exp_files[fn.replace('.json', '')] = 1

        self._load_data_maml()

        if cfg.limit_bspn_vocab:
            self.bspn_masks = self._construct_bspn_constraint()
        if cfg.limit_aspn_vocab:
            self.aspn_masks = self._construct_aspn_constraint()

        self.multi_acts_record = None

    def _build_vocab(self):
        self.vocab = utils.Vocab(cfg.vocab_size)
        vp = cfg.vocab_path_train if cfg.mode == 'train' or cfg.vocab_path_eval is None else cfg.vocab_path_eval
        # vp = cfg.vocab_path+'.json.freq.json'
        self.vocab.load_vocab(vp)
        return self.vocab.vocab_size

    def _construct_bspn_constraint(self):
        bspn_masks = {}
        valid_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']
        all_dom_codes = [self.vocab.encode('['+d+']') for d in valid_domains]
        all_slot_codes = [self.vocab.encode(s) for s in ontology.all_slots]
        bspn_masks[self.vocab.encode('<go_b>')] = all_dom_codes + [self.vocab.encode('<eos_b>'), 0]
        bspn_masks[self.vocab.encode('<eos_b>')] = [self.vocab.encode('<pad>')]
        bspn_masks[self.vocab.encode('<pad>')] = [self.vocab.encode('<pad>')]
        for domain, slot_values in self.slot_value_set.items():
            if domain == 'police':
                continue
            dom_code = self.vocab.encode('['+domain+']')
            bspn_masks[dom_code] = []
            for slot, values in slot_values.items():
                slot_code = self.vocab.encode(slot)
                if slot_code not in bspn_masks:
                    bspn_masks[slot_code] = []
                if slot_code not in bspn_masks[dom_code]:
                    bspn_masks[dom_code].append(slot_code)
                for value in values:
                    for idx, v in enumerate(value.split()):
                        if not self.vocab.has_word(v):
                            continue
                        v_code = self.vocab.encode(v)
                        if v_code not in bspn_masks:
                            # print(self.vocab._word2idx)
                            bspn_masks[v_code] = []
                        if idx == 0 and v_code not in bspn_masks[slot_code]:
                            bspn_masks[slot_code].append(v_code)
                        if idx == (len(value.split()) - 1):
                            for w in all_dom_codes + all_slot_codes:
                                if self.vocab.encode('<eos_b>') not in bspn_masks[v_code]:
                                    bspn_masks[v_code].append(self.vocab.encode('<eos_b>'))
                                if w not in bspn_masks[v_code]:
                                    bspn_masks[v_code].append(w)
                            break
                        if not self.vocab.has_word(value.split()[idx + 1]):
                            continue
                        next_v_code = self.vocab.encode(value.split()[idx + 1])
                        if next_v_code not in bspn_masks[v_code]:
                            bspn_masks[v_code].append(next_v_code)
        bspn_masks[self.vocab.encode('<unk>')] = list(bspn_masks.keys())

        with open('data/multi-woz-processed/bspn_masks.txt', 'w') as f:
            for i,j in bspn_masks.items():
                f.write(self.vocab.decode(i) + ': ' + ' '.join([self.vocab.decode(int(m)) for m in j])+'\n')
        return bspn_masks

    def _construct_aspn_constraint(self):
        aspn_masks = {}
        aspn_masks = {}
        all_dom_codes = [self.vocab.encode('['+d+']') for d in ontology.dialog_acts.keys()]
        all_act_codes = [self.vocab.encode('['+a+']') for a in ontology.dialog_act_params]
        all_slot_codes = [self.vocab.encode(s) for s in ontology.dialog_act_all_slots]
        aspn_masks[self.vocab.encode('<go_a>')] = all_dom_codes + [self.vocab.encode('<eos_a>'), 0]
        aspn_masks[self.vocab.encode('<eos_a>')] = [self.vocab.encode('<pad>')]
        aspn_masks[self.vocab.encode('<pad>')] = [self.vocab.encode('<pad>')]
        # for d in all_dom_codes:
        #     aspn_masks[d] = all_act_codes
        for a in all_act_codes:
            aspn_masks[a] = all_dom_codes + all_slot_codes + [self.vocab.encode('<eos_a>')]
        for domain, acts in ontology.dialog_acts.items():
            dom_code = self.vocab.encode('['+domain+']')
            aspn_masks[dom_code] = []
            for a in acts:
                act_code = self.vocab.encode('['+a+']')
                if act_code not in aspn_masks[dom_code]:
                    aspn_masks[dom_code].append(act_code)
        # for a, slots in ontology.dialog_act_params.items():
        #     act_code = self.vocab.encode('['+a+']')
        #     slot_codes = [self.vocab.encode(s) for s in slots]
        #     aspn_masks[act_code] = all_dom_codes + slot_codes + [self.vocab.encode('<eos_a>')]
        for s in all_slot_codes:
            aspn_masks[s] = all_dom_codes + all_slot_codes + [self.vocab.encode('<eos_a>')]
        aspn_masks[self.vocab.encode('<unk>')] = list(aspn_masks.keys())


        with open('data/multi-woz-processed/aspn_masks.txt', 'w') as f:
            for i,j in aspn_masks.items():
                f.write(self.vocab.decode(i) + ': ' + ' '.join([self.vocab.decode(int(m)) for m in j])+'\n')
        return aspn_masks

    def _load_data(self, save_temp=False):
        self.data = json.loads(open(cfg.data_path+cfg.data_file, 'r', encoding='utf-8').read().lower())
        self.train, self.dev, self.test = [] , [], []
        for fn, dial in self.data.items():
            if 'all' in cfg.exp_domains or self.exp_files.get(fn):
                if self.dev_files.get(fn):
                    self.dev.append(self._get_encoded_data(fn, dial))
                elif self.test_files.get(fn):
                    self.test.append(self._get_encoded_data(fn, dial))
                else:
                    self.train.append(self._get_encoded_data(fn, dial))
        if save_temp:
            json.dump(self.test, open('data/multi-woz-analysis/test.encoded.json','w'), indent=2)
            self.vocab.save_vocab('data/multi-woz-analysis/vocab_temp')

        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)    

    def _load_data_maml(self, save_temp=False):
        self.data = {}
        for domain in cfg.train_data_file:
            data_json = open(os.path.join(cfg.data_path + domain), 'r', encoding='utf-8')
            data_dom = json.loads(data_json.read().lower())
            data_json.close()

            train_num = len(data_dom) * cfg.split[0] // sum(cfg.split)
            train_list = random.choices(list(data_dom.keys()), k = train_num)
            train_data, dev_data = [], []
            for fn, dial in data_dom.items():
                if fn in train_list:
                    train_data.append(self._get_encoded_data(fn, dial))
                else:
                    dev_data.append(self._get_encoded_data(fn, dial))

            random.shuffle(train_data)
            random.shuffle(dev_data)

            self.train.append(train_data)
            self.dev.append(dev_data)

            self.data.update(data_dom)

        test_data_json = open(os.path.join(cfg.data_path + cfg.test_data_file), 'r', encoding='utf-8')
        test_data_dom = json.loads(test_data_json.read().lower())
        test_data_json.close()
        adapt_data_json = open(os.path.join(cfg.data_path + cfg.adapt_data_file), 'r', encoding='utf-8')
        adapt_data_dom = json.loads(adapt_data_json.read().lower())
        adapt_data_json.close()

        self.test = [self._get_encoded_data(fn, dial) for (fn, dial) in test_data_dom.items()]
        self.adapt = [self._get_encoded_data(fn, dial) for (fn, dial) in adapt_data_dom.items()]
        if save_temp:
            json.dump(self.test, open('data/multi-woz-analysis/test.encoded.json','w'), indent = 4)
            self.vocab.save_vocab('data/multi-woz-analysis/vocab_temp')
            
        random.shuffle(self.test)
        random.shuffle(self.adapt)

        self.data.update(test_data_dom)
        self.data.update(adapt_data_dom)

    def _get_encoded_data(self, fn, dial):
        encoded_dial = []
        for idx, t in enumerate(dial['log']):
            enc = {}
            enc['dial_id'] = fn
            enc['user'] = self.vocab.sentence_encode(t['user'].split() + ['<eos_u>'])
            enc['usdx'] = self.vocab.sentence_encode(t['user_delex'].split() + ['<eos_u>'])
            enc['resp'] = self.vocab.sentence_encode(t['resp'].split() + ['<eos_r>'])
            enc['bspn'] = self.vocab.sentence_encode(t['constraint'].split() + ['<eos_b>'])
            enc['bsdx'] = self.vocab.sentence_encode(t['cons_delex'].split() + ['<eos_b>'])
            enc['aspn'] = self.vocab.sentence_encode(t['sys_act'].split() + ['<eos_a>'])
            enc['dspn'] = self.vocab.sentence_encode(t['turn_domain'].split() + ['<eos_d>'])
            enc['pointer'] = [int(i) for i in t['pointer'].split(',')]
            enc['turn_domain'] = t['turn_domain'].split()
            enc['turn_num'] = t['turn_num']
            if cfg.token_weight > 0 :
                if 'mixed_probs_resp_' + str(cfg.token_weight) not in t:
                    pdb.set_trace()
                enc['token_weight'] = t['mixed_probs_resp_' + str(cfg.token_weight)]

            if cfg.multi_acts_training:
                enc['aspn_aug'] = []
                if fn in self.multi_acts:
                    turn_ma = self.multi_acts[fn].get(str(idx), {})
                    for act_type, act_spans in turn_ma.items():
                        enc['aspn_aug'].append([self.vocab.sentence_encode(a.split()+['<eos_a>']) for a in act_spans])

            encoded_dial.append(enc)
        return encoded_dial

    def bspan_to_constraint_dict(self, bspan, bspn_mode = 'bspn'):
        bspan = bspan.split() if isinstance(bspan, str) else bspan
        constraint_dict = {}
        domain = None
        conslen = len(bspan)
        for idx, cons in enumerate(bspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons == '<eos_b>':
                break
            if '[' in cons:
                if cons[1:-1] not in ontology.all_domains:
                    continue
                domain = cons[1:-1]
            elif cons in ontology.get_slot:
                if domain is None:
                    continue
                if cons == 'people':
                    # handle confusion of value name "people's portraits..." and slot people
                    try:
                        ns = bspan[idx+1]
                        ns = self.vocab.decode(ns) if type(ns) is not str else ns
                        if ns == "'s":
                            continue
                    except:
                        continue
                if not constraint_dict.get(domain):
                    constraint_dict[domain] = {}
                if bspn_mode == 'bsdx':
                    constraint_dict[domain][cons] = 1
                    continue
                vidx = idx+1
                if vidx == conslen:
                    break
                vt_collect = []
                vt = bspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                while vidx < conslen and vt != '<eos_b>' and '[' not in vt and vt not in ontology.get_slot:
                    vt_collect.append(vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = bspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if vt_collect:
                    constraint_dict[domain][cons] = ' '.join(vt_collect)

        return constraint_dict

    def bspan_to_DBpointer(self, bspan, turn_domain):
        constraint_dict = self.bspan_to_constraint_dict(bspan)
        # print(constraint_dict)
        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith('[') else match_dom
        match = matnums[match_dom]
        vector = self.db.addDBPointer(match_dom, match)
        return vector

    def aspan_to_act_list(self, aspan):
        aspan = aspan.split() if isinstance(aspan, str) else aspan
        acts = []
        domain = None
        conslen = len(aspan)
        for idx, cons in enumerate(aspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons == '<eos_a>':
                break
            if '[' in cons and cons[1:-1] in ontology.dialog_acts:
                domain = cons[1:-1]

            elif '[' in cons and cons[1:-1] in ontology.dialog_act_params:
                if domain is None:
                    continue
                vidx = idx+1
                if vidx == conslen:
                    acts.append(domain+'-'+cons[1:-1]+'-none')
                    break
                vt = aspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                no_param_act = True
                while vidx < conslen and vt != '<eos_a>' and '[' not in vt:
                    no_param_act = False
                    acts.append(domain+'-'+cons[1:-1]+'-'+vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = aspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if no_param_act:
                    acts.append(domain+'-'+cons[1:-1]+'-none')

        return acts

    def dspan_to_domain(self, dspan):
        domains = {}
        dspan = dspan.split() if isinstance(dspan, str) else dspan
        for d in dspan:
            dom = self.vocab.decode(d) if type(d) is not str else d
            if dom != '<eos_d>':
                domains[dom] = 1
            else:
                break
        return domains

    def convert_batch(self, py_batch, py_prev, first_turn=False):
        inputs = {}
        if first_turn:
            for item, py_list in py_prev.items():
                batch_size = len(py_batch['user'])
                inputs[item+'_np'] = np.array([[1]] * batch_size)
                inputs[item+'_unk_np'] = np.array([[1]] * batch_size)
        else:
            for item, py_list in py_prev.items():
                if py_list is None:
                    continue
                if not cfg.enable_aspn and 'aspn' in item:
                    continue
                if not cfg.enable_bspn and 'bspn' in item:
                    continue
                if not cfg.enable_dspn and 'dspn' in item:
                    continue
                prev_np = utils.padSeqs(py_list, truncated=cfg.truncated, trunc_method='pre')
                inputs[item+'_np'] = prev_np
                if item in ['pv_resp', 'pv_bspn']:
                    inputs[item+'_unk_np'] = deepcopy(inputs[item+'_np'])
                    inputs[item+'_unk_np'][inputs[item+'_unk_np']>=self.vocab_size] = 2   # <unk>
                else:
                    inputs[item+'_unk_np'] = inputs[item+'_np']

        for item in ['user', 'usdx', 'resp', 'bspn', 'aspn', 'bsdx', 'dspn']:
            if not cfg.enable_aspn and item == 'aspn':
                continue
            if not cfg.enable_bspn and item == 'bspn':
                continue

            if not cfg.enable_dspn and item == 'dspn':
                continue
            py_list = py_batch[item]
            trunc_method = 'post' if item == 'resp' else 'pre'
            # max_length = cfg.max_nl_length if item in ['user', 'usdx', 'resp'] else cfg.max_span_length
            inputs[item+'_np'] = utils.padSeqs(py_list, truncated=cfg.truncated, trunc_method=trunc_method)
            if item in ['user', 'usdx', 'resp', 'bspn']:
                inputs[item+'_unk_np'] = deepcopy(inputs[item+'_np'])
                inputs[item+'_unk_np'][inputs[item+'_unk_np']>=self.vocab_size] = 2   # <unk>
            else:
                inputs[item+'_unk_np'] = inputs[item+'_np']

        if cfg.multi_acts_training and cfg.mode=='train':
            inputs['aspn_bidx'], multi_aspn = [], []
            for bidx, aspn_type_list in enumerate(py_batch['aspn_aug']):
                if aspn_type_list:
                    for aspn_list in aspn_type_list:
                        random.shuffle(aspn_list)
                        aspn = aspn_list[0]   #choose one random act span in each act type
                        multi_aspn.append(aspn)
                        inputs['aspn_bidx'].append(bidx)
                        if cfg.multi_act_sampling_num>1:
                            for i in range(cfg.multi_act_sampling_num):
                                if len(aspn_list) >= i+2:
                                    aspn = aspn_list[i+1]   #choose one random act span in each act type
                                    multi_aspn.append(aspn)
                                    inputs['aspn_bidx'].append(bidx)

            if multi_aspn:
                inputs['aspn_aug_np'] = utils.padSeqs(multi_aspn, truncated=cfg.truncated, trunc_method='pre')
                inputs['aspn_aug_unk_np'] = inputs['aspn_aug_np']   # [all available aspn num in the batch, T]

        inputs['db_np'] = np.array(py_batch['pointer'])
        inputs['turn_domain'] = py_batch['turn_domain']

        if 'token_weight' in py_batch:
            # # # padding weight
            inputs['token_weight'] = deepcopy(py_batch['token_weight'])
            # pdb.set_trace()
            max_length = max([len(sent) for sent in inputs['token_weight']])
            # if max_length != len(inputs['resp_np'][0]):
            #     pdb.set_trace()

            for idx in range(len(inputs['token_weight'])):
                while len(inputs['token_weight'][idx]) < max_length:
                    inputs['token_weight'][idx].append(0)

            # for i in range(len(inputs['token_weight'])):
            #     if len(inputs['token_weight'][i]) != len(inputs['resp_np'][i]):
            #         pdb.set_trace()
        return inputs

    def wrap_result(self, result_dict, eos_syntax=None):
        decode_fn = self.vocab.sentence_decode
        results = []
        eos_syntax = ontology.eos_tokens if not eos_syntax else eos_syntax

        if cfg.bspn_mode == 'bspn':
            field = ['dial_id', 'turn_num', 'user', 'bspn_gen','bspn', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                        'dspn_gen', 'dspn', 'pointer']
        elif not cfg.enable_dst:
            field = ['dial_id', 'turn_num', 'user', 'bsdx_gen','bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                        'dspn_gen', 'dspn', 'bspn', 'pointer']
        else:
            field = ['dial_id', 'turn_num', 'user', 'bsdx_gen','bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                        'dspn_gen', 'dspn', 'bspn_gen','bspn', 'pointer']
        if self.multi_acts_record is not None:
            field.insert(7, 'multi_act_gen')

        if cfg.token_weight == -1:
            field.append('token_weight')

        for dial_id, turns in result_dict.items():
            entry = {'dial_id': dial_id, 'turn_num': len(turns)}

            for prop in field[2:]:
                entry[prop] = ''
            results.append(entry)

            # pdb.set_trace()
            for turn_no, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    if key == 'turn_domain':
                        v = ' '.join(v)
                    entry[key] = decode_fn(v, eos=eos_syntax[key]) if key in eos_syntax and v != '' else v
                results.append(entry)
        return results, field

    def restore(self, resp, domain, constraint_dict, mat_ents):
        restored = resp

        restored = restored.replace('[value_reference]', '53022')
        restored = restored.replace('[value_car]', 'BMW')

        # restored.replace('[value_phone]', '830-430-6666')
        for d in domain:
            constraint = constraint_dict.get(d,None)
            if constraint:
                if 'stay' in constraint:
                    restored = restored.replace('[value_stay]', constraint['stay'])
                if 'day' in constraint:
                    restored = restored.replace('[value_day]', constraint['day'])
                if 'people' in constraint:
                    restored = restored.replace('[value_people]', constraint['people'])
                if 'time' in constraint:
                    restored = restored.replace('[value_time]', constraint['time'])
                if 'type' in constraint:
                    restored = restored.replace('[value_type]', constraint['type'])
                if d in mat_ents and len(mat_ents[d])==0:
                    for s in constraint:
                        if s == 'pricerange' and d in ['hotel', 'restaurant'] and 'price]' in restored:
                            restored = restored.replace('[value_price]', constraint['pricerange'])
                        if s+']' in restored:
                            restored = restored.replace('[value_%s]'%s, constraint[s])

            if '[value_choice' in restored and mat_ents.get(d):
                restored = restored.replace('[value_choice]', str(len(mat_ents[d])))
        if '[value_choice' in restored:
            restored = restored.replace('[value_choice]', '3')


        # restored.replace('[value_car]', 'BMW')


        try:
            ent = mat_ents.get(domain[-1], [])
            if ent:
                ent = ent[0]

                for t in restored.split():
                    if '[value' in t:
                        slot = t[7:-1]
                        if ent.get(slot):
                            if domain[-1] == 'hotel' and slot == 'price':
                                slot = 'pricerange'
                            restored = restored.replace(t, ent[slot])
                        elif slot == 'price':
                            if ent.get('pricerange'):
                                restored = restored.replace(t, ent['pricerange'])
                            else:
                                print(restored, domain)
        except:
            print(resp)
            print(restored)
            quit()


        restored = restored.replace('[value_phone]', '62781111')
        restored = restored.replace('[value_postcode]', 'CG9566')
        restored = restored.replace('[value_address]', 'Parkside, Cambridge')

        # if '[value_' in restored:

        #     print(domain)
        #     # print(mat_ents)
        #     print(resp)
        #     print(restored)
        return restored

    def record_utterance(self, result_dict):
        decode_fn = self.vocab.sentence_decode


        ordered_dial = {}
        for dial_id, turns in result_dict.items():
            diverse = 0
            turn_count = 0
            for turn_no, turn in enumerate(turns):
                act_collect = {}
                act_type_collect = {}
                slot_score = 0
                for i in range(cfg.nbest):
                    aspn = decode_fn(turn['multi_act'][i], eos=ontology.eos_tokens['aspn'])
                    pred_acts = self.aspan_to_act_list(' '.join(aspn))
                    act_type = ''
                    for act in pred_acts:
                        d,a,s = act.split('-')
                        if d + '-' + a not in act_collect:
                            act_collect[d + '-' + a] = {s:1}
                            slot_score += 1
                            act_type += d + '-' + a + ';'
                        elif s not in act_collect:
                            act_collect[d + '-' + a][s] = 1
                            slot_score += 1
                    act_type_collect[act_type] = 1
                turn_count += 1
                diverse += len(act_collect) * 3 + slot_score
            ordered_dial[dial_id] = diverse/turn_count

        ordered_dial = sorted(ordered_dial.keys(), key=lambda x: -ordered_dial[x])



        dialog_record = {}

        with open(cfg.eval_load_path + '/dialogue_record.csv', 'w') as rf:
            writer = csv.writer(rf)


            for dial_id in ordered_dial:
                dialog_record[dial_id] = []
                turns = result_dict[dial_id]
                writer.writerow([dial_id])
                for turn_no, turn in enumerate(turns):
                    user =decode_fn(turn['user'], eos=ontology.eos_tokens['user'])
                    bspn = decode_fn(turn['bspn'], eos=ontology.eos_tokens['bspn'])
                    aspn = decode_fn(turn['aspn'], eos=ontology.eos_tokens['aspn'])
                    resp = decode_fn(turn['resp'], eos=ontology.eos_tokens['resp'])
                    constraint_dict = self.bspan_to_constraint_dict(bspn)
                    # print(constraint_dict)
                    mat_ents = self.db.get_match_num(constraint_dict, True)
                    domain = [i[1:-1] for i in self.dspan_to_domain(turn['dspn']).keys()]
                    restored = self.restore(resp, domain, constraint_dict, mat_ents)
                    writer.writerow([turn_no, user, turn['pointer'], domain, restored, resp ])
                    turn_record = {'user':user, 'bspn': bspn, 'aspn':aspn, 'dom':domain, 'resp':resp, 'resp_res':restored}

                    resp_col = []
                    aspn_col = []
                    resp_restore_col = []
                    for i in range(cfg.nbest):
                        aspn = decode_fn(turn['multi_act'][i], eos=ontology.eos_tokens['aspn'])
                        resp = decode_fn(turn['multi_resp'][i], eos=ontology.eos_tokens['resp'])


                        restored = self.restore(resp, domain, constraint_dict, mat_ents)
                        resp_col.append(resp)
                        resp_restore_col.append(restored)
                        aspn_col.append(aspn)


                    zipped = list(zip(resp_restore_col, resp_col, aspn_col))
                    zipped.sort(key = lambda s: len(s[0]))
                    resp_restore_col = list(list(zip(*zipped))[0])
                    aspn_col = list(list(zip(*zipped))[2])
                    resp_col = list(list(zip(*zipped))[1])
                    turn_record['aspn_col'] = aspn_col
                    turn_record['resp_col'] = resp_col
                    turn_record['resp_res_col'] = resp_restore_col
                    for i in range(cfg.nbest):
                        # aspn = decode_fn(turn['multi_act'][i], eos=ontology.eos_tokens['aspn'])
                        resp = resp_col[i]
                        aspn = aspn_col[i]
                        resp_restore = resp_restore_col[i]

                        writer.writerow(['',resp_restore, resp, aspn])

                    dialog_record[dial_id].append(turn_record)

            # json.dump(dialog_record, open(cfg.eval_load_path + '/resultdict.json','w'))


if __name__=='__main__':
    reader = MultiWozReader()
    # for aspan in ["[general] [bye] [welcome] <eos_a>","[train] [inform] trainid destination arrive leave [offerbook] [general] [reqmore] <eos_a>",]:
    #     act = reader.aspan_to_constraint_dict(aspan.split())
    #     print('！！！')
    #     print(act)

    for bspan in ["[taxi] destination golden house departure broughton house gallery arrive 19:30 [attraction] type museum name whipple museum of the history of science people 5 day monday", "[taxi] destination golden house departure broughton house gallery arrive 19:30 [attraction] type museum name whipple museum of the history of science people 5 day monday <eos_b>"]:
        encoded=reader.vocab.sentence_encode(bspan.split())
        print(encoded)
        cons = reader.bspan_to_constraint_dict(encoded, bspn_mode='bspn')
        print(cons)
    for bspan in  ["[taxi] destination departure leave [hotel] name [attraction] name people day", "[taxi] destination departure leave [hotel] name [attraction] name people day <eos_b>"]:
        encoded=reader.vocab.sentence_encode(bspan.split())
        print(encoded)
        cons = reader.bspan_to_constraint_dict(encoded, bspn_mode='bsdx')
        print(cons)

