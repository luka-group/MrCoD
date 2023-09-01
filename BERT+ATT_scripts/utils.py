import json
import redis
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer


def get_passage_span(doc, pid):
    token_start = doc["passage_mapping"][pid-1] if pid!= 0 else 0
    token_end = doc["passage_mapping"][pid]
    return token_start, token_end


def get_sent_span(doc, pid, sid):
    if sid == 0:
        if pid != 0:
            token_start = doc["sentence_mapping"][pid-1][-1]
        else:
            token_start = 0
    else:
        token_start = doc["sentence_mapping"][pid][sid-1]
    token_end = doc["sentence_mapping"][pid][sid]
    return token_start, token_end


def get_entity_span_by_pid(doc, pid):
    entities = defaultdict(list)
    for ent in doc['entities']:
        for span in ent['spans']:
            if 'Q' in ent and span[2] == pid:
                entities[("Q"+str(ent['Q']))].append(span)
    return entities


def get_entity_span_by_range(doc, token_range):
    begin, end = token_range
    entities = defaultdict(list)
    for ent in doc['entities']:
        for span in ent['spans']:
            if 'Q' in ent and begin <= span[0] and span[1] <= end:
                entities[("Q"+str(ent['Q']))].append(span)
    return entities


def get_sent_mapping(doc, index, bridging_ents):
    sent_map = defaultdict(list)
    for pid in index:
        for key, spans in get_entity_span_by_pid(doc, pid).items():
            for span in spans:
                if key in bridging_ents:
                    sent_map[(span[2], span[3])].append(key)
    return sent_map


def get_bridging_ents(h, t, doch, doct, index_h, index_t):
    ent_h = [get_entity_span_by_pid(doch, pid) for pid in index_h]
    ent_t = [get_entity_span_by_pid(doct, pid) for pid in index_t]

    ent_h_names = defaultdict(lambda: 0)
    for d in ent_h:
        for key in d:
            ent_h_names[key] += 1
    ent_t_names = defaultdict(lambda: 0)
    for d in ent_t:
        for key in d:
            ent_t_names[key] += 1

    bridges = set()
    for key in ent_h_names:
        if ent_h_names[key] >= 2:
            bridges.add(key)
    for key in ent_t_names:
        if ent_t_names[key] >= 2:
            bridges.add(key)

    bridging_ents = set([h, t]).union(bridges)
    return bridging_ents


def merge_interval(arr):
    arr.sort(key=lambda x: x[0])
    index = 0
    for i in range(1, len(arr)):
        if (arr[index][1] >= arr[i][0]):
            arr[index][1] = max(arr[index][1], arr[i][1])
        else:
            index = index + 1
            arr[index] = arr[i]
    return arr[:(index+1)]


def get_token_span(h, t, doch, doct, index_h, index_t, bridging_ents, max_length=512):
    sent_map_h = get_sent_mapping(doch, index_h, bridging_ents)
    sent_map_t = get_sent_mapping(doct, index_t, bridging_ents)

    index_h_token = [get_sent_span(doch, pid, sid) for pid, sid in sent_map_h.keys()]
    index_t_token = [get_sent_span(doct, pid, sid) for pid, sid in sent_map_t.keys()]
    length = sum([end - start for start, end in (index_h_token + index_t_token)])

    sent_num = len(index_h_token + index_t_token)

    max_h, max_t = len(doch['tokens']), len(doct['tokens'])

    if length < max_length:
        expand = 0
        res_span = (max_length - length) // sent_num // 2
        while expand < 5 and res_span > 5 and length < max_length:
            index_h_token = [
                [max(0, start - res_span), min(max_h, end + res_span)] for start, end in index_h_token]
            index_h_token = merge_interval(index_h_token)

            index_t_token = [
                [max(0, start - res_span), min(max_t, end + res_span)] for start, end in index_t_token]
            index_t_token = merge_interval(index_t_token)

            length = sum([end - start for start, end in (index_h_token + index_t_token)])
            res_span = (max_length - length) // sent_num // 2

            expand += 1
    else:
        target_length = max_length // sent_num

        spare_length = 0
        spare_sent_num = 0
        for start, end in (index_h_token + index_t_token):
            p_length = end - start
            if p_length < target_length:
                spare_length += target_length - p_length
                spare_sent_num += 1
        adj_target_length = target_length + spare_length // (sent_num - spare_sent_num)

        for i in range(len(index_h_token)):
            if index_h_token[i][1] - index_h_token[i][0] > target_length:
                ent_span = get_entity_span_by_range(doch, index_h_token[i])
                if h in ent_span:
                    h_start, h_end, _, _ = ent_span[h][0]
                    ent_length = h_end - h_start
                    res_span = (adj_target_length - ent_length) // 2
                    index_h_token[i] = (
                        max(index_h_token[i][0], h_start - res_span),
                        min(index_h_token[i][1], h_end + res_span)
                        )
                else:
                    index_h_token[i] = (index_h_token[i][0], index_h_token[i][0] + adj_target_length)

        for i in range(len(index_t_token)):
            if index_t_token[i][1] - index_t_token[i][0] > target_length:
                ent_span = get_entity_span_by_range(doct, index_t_token[i])
                if t in ent_span:
                    t_start, t_end, _, _ = ent_span[t][0]
                    ent_length = t_end - t_start
                    res_span = (adj_target_length - ent_length) // 2
                    index_t_token[i] = (
                        max(index_t_token[i][0], t_start - res_span),
                        min(index_t_token[i][1], t_end + res_span)
                        )
                else:
                    index_t_token[i] = (index_t_token[i][0], index_t_token[i][0] + adj_target_length)

    length = sum([end - start for start, end in (index_h_token + index_t_token)])

    return index_h_token, index_t_token, length


def get_tokens_and_entities(index_token, doc):
    tokens = []
    entities = []
    for start, end in index_token:
        current_length = len(tokens)
        tokens += doc['tokens'][start:end]
        for ent in doc['entities']:
            for s, e, pid, sid in ent['spans']:
                if s >= start and e <= end and 'Q' in ent:
                    entities.append({
                        "id": "Q" + str(ent['Q']),
                        "name": ent['name'],
                        "span": [s - start + current_length, e - start + current_length],
                        "type": ent['type']
                        })
    return tokens, entities


def get_context(tokens, entities, ma, add_special=False):
    #span_start = {each["span"][0]: each["id"] for each in entities}
    #span_end = {each["span"][1]: each["id"] for each in entities}
    span_start = defaultdict(list)
    span_end = defaultdict(list)
    for each in entities:
        span_start[each["span"][0]].append(each["id"])
        span_end[each["span"][1]].append(each["id"])

    token_output = []
    for i, token in enumerate(tokens):
        if i in span_end:
            for eid in span_end[i]:
                if eid in ma:
                    idx = ma[eid]*2+2
                    if idx == 2 or idx == 4:
                        token_output.append("[unused{ID}]".format(ID=idx))
                    else:
                        if add_special:
                            token_output.append("[unused{ID}]".format(ID=idx))
        if i in span_start:
            for eid in span_start[i]:
                if eid in ma:
                    idx = ma[eid]*2+1
                    if idx == 1 or idx == 3:
                        token_output.append("[unused{ID}]".format(ID=idx))
                    else:
                        if add_special:
                            token_output.append("[unused{ID}]".format(ID=idx))
        token_output.append(token)

    n = len(tokens)
    if n in span_end:
        for eid in span_end[n]:
            if eid in ma:
                idx = ma[eid]*2+2
                if idx == 2 or idx == 4:
                    token_output.append("[unused{ID}]".format(ID=idx))
                else:
                    if add_special:
                        token_output.append("[unused{ID}]".format(ID=idx))
    return token_output

def get_passage_context(h, t,
   doch_title,
   doct_title,
   tokenizer,
   redisd,
   retrieval_index,
   max_length = 500,
   expand = True,
   bridge_only = True,
   add_special=False):

    max_length -= 5
    key = "|".join([h, t, doch_title, doct_title])

    index_h, index_t = retrieval_index[key]
    doch = json.loads(redisd.get("codred-doc-open-%s" % doch_title))
    doct = json.loads(redisd.get("codred-doc-open-%s" % doct_title))

    bridging_ents = get_bridging_ents(h, t, doch, doct, index_h, index_t)

    index_h_token, index_t_token, _ = get_token_span(h, t, doch, doct, index_h, index_t, bridging_ents, max_length)

    doch_tokens, doch_entities = get_tokens_and_entities(index_h_token, doch)
    doct_tokens, doct_entities = get_tokens_and_entities(index_t_token, doct)

    ma = {h: 0, t: 1}
    for e in (doch_entities + doct_entities):
      if bridge_only:
         if e["id"] in bridging_ents:
            if e["id"] not in ma:
               ma[e["id"]] = len(ma)
      else:
         if e["id"] not in ma:
            ma[e["id"]] = len(ma)

    token_h = get_context(doch_tokens, doch_entities, ma, add_special)
    token_t = get_context(doct_tokens, doct_entities, ma, add_special)

    final_length = len(token_h) + len(token_t)
    if final_length > max_length:
        if len(token_h) > len(token_t):
            token_h = token_h[:(max_length - len(token_t))]
        else:
            token_t = token_t[:(max_length - len(token_h))]

    tokens = ['[CLS]'] + token_h + ['[SEP]'] + token_t + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    max_length += 5

    if len(token_ids) < max_length:
        token_ids = token_ids + [0] * (max_length - len(tokens))
    attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
    token_type_id = [0] * (len(token_h) + 2) + [1] * (len(token_t) + 1) + [0] * (max_length - len(tokens))

    return tokens, token_ids, token_type_id, attention_mask


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
    redisd = redis.Redis(host='localhost', port=6379, decode_responses=True)

    retrieval_index = json.load(open("../data/retrieval_index/path_mining_3_hop_finetune_ext_inference_open_shared_entities.json"))

    key = list(retrieval_index.keys())[34]
    h, t, doch_title, doct_title = key.split("|")
    index_h, index_t = retrieval_index[key]
    tokens, token_ids, _, _  = get_passage_context(h, t, doch_title, doct_title, tokenizer, redisd, retrieval_index)
    string = tokenizer.convert_tokens_to_string(tokens)
    print(token_ids, string)
