import os
from collect_data_scripts.utils import *
import json


def collect_hq_data_path():
    data_map = load_data_map()



def load_data_map():
    data_map_path = "m2_pt_data_map_0731.txt"
    data_map = {}
    with open(data_map_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            dataset_name = parts[0]
            dataset_path_str = parts[1]
            file_path_list = get_all_jsonl_files(dataset_path_str)
            hq_file_path_list = filter_hq_path(file_path_list)
            data_map[dataset_name] = {"dataset_path_str": dataset_path_str,
                                      "total_file_num": len(file_path_list),
                                      "hq_file_num": len(hq_file_path_list),
                                      "file_path_list": file_path_list,
                                      "hq_file_path_list": hq_file_path_list}
    return data_map


def filter_hq_path(file_path_list):
    res = []
    exclude_patterns = {"MH", "ML", "LQ", "ELQ"}
    for file_path in file_path_list:
        for pattern in exclude_patterns:
            if "_" + pattern in file_path:
                continue
            res.append(file_path)
    return res


def test():
    res = load_data_map()
    with open("local_data/test_data_0731/collect_sta_data_0731.json", "w") as fo:
        fo.write(json.dumps(res, indent=2))


if __name__ == "__main__":
    test()


