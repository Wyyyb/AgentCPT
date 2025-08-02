import os
import json
import random


def sample_100_each_dataset():
    res_data = []
    with open("../local_data/test_data_0731/collect_sta_data_0731_full.json", "r") as fi:
        data_map = json.load(fi)
    for k, v in data_map.items():
        print("dataset name", k)
        hq_file_path_list = v["hq_file_path_list"]
        random.shuffle(hq_file_path_list)
        with open(hq_file_path_list[0], "r") as fi:
            for line in fi.readlines():
                curr = json.loads(line)
                curr["agent_cpt_dict"] = {"file_path": hq_file_path_list[0],
                                          "dataset_name": k}
                res_data.append(curr)
    with open("../local_data/test_data_0731/sample_100_each_data.json", "w") as fo:
        fo.write(json.dumps(res_data, indent=4))


if __name__ == '__main__':
    sample_100_each_dataset()


