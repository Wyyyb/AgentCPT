import os
from utils import *
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
            if check_duplicate_list(file_path_list):
                print("Duplicate", dataset_name)
            hq_file_path_list = filter_hq_path(file_path_list)
            total_line_num = 0
            hq_line_num = 0
            for each in file_path_list:
                total_line_num += count_lines_jsonl(each)
            for each in hq_file_path_list:
                hq_line_num += count_lines_jsonl(each)
            data_map[dataset_name] = {"dataset_path_str": dataset_path_str,
                                      "total_file_num": len(file_path_list),
                                      "hq_file_num": len(hq_file_path_list),
                                      "file_path_list": file_path_list,
                                      "hq_file_path_list": hq_file_path_list,
                                      "total_line_num": total_line_num,
                                      "hq_line_num": hq_line_num}
    return data_map


def filter_hq_path(file_path_list):
    res = []
    exclude_patterns = {"MH", "ML", "LQ", "ELQ", "low_quality"}
    for file_path in file_path_list:
        flag = 0
        for pattern in exclude_patterns:
            if "_" + pattern in file_path:
                flag = 1
                # print("detect exclude_patterns", file_path)
                break
        if flag == 0:
            res.append(file_path)
    return res


def sta_collect_data(res):
    sta_res = []
    for k, v in res.items():
        dataset_name = k
        total_file_num = str(v["total_file_num"])
        hq_file_num = str(v["hq_file_num"])
        total_line_num = str(v["total_line_num"])
        hq_line_num = str(v["hq_line_num"])
        # sta_res.append(f"{dataset_name}\t{total_file_num}\t{hq_file_num}\t{total_line_num}\t{hq_line_num}\n")
        sta_res.append(
            "{}\t{}\t{}\t{}\t{}\n".format(dataset_name, total_file_num, hq_file_num, total_line_num, hq_line_num))
    with open("../local_data/test_data_0731/collect_sta_data_0731.txt", "w") as fo:
        fo.write("".join(sta_res))


def test():
    res = load_data_map()
    sta_collect_data(res)
    with open("../local_data/test_data_0731/collect_sta_data_0731_full.json", "w") as fo:
        fo.write(json.dumps(res, indent=2))


if __name__ == "__main__":
    test()


