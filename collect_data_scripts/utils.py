import os
import glob


def get_all_jsonl_files(path_string):
    all_jsonl_files = []

    paths = path_string.split(',')

    for path in paths:
        path = path.strip()

        if path.endswith('.jsonl') and ('*' in path or '?' in path):
            matching_files = glob.glob(path)
            all_jsonl_files.extend(matching_files)

        else:
            if '*' in path or '?' in path:
                matching_dirs = glob.glob(path)
            else:
                matching_dirs = [path]

            for directory in matching_dirs:
                if os.path.isdir(directory):
                    for root, _, files in os.walk(directory):
                        for file in files:
                            if file.endswith('.jsonl'):
                                all_jsonl_files.append(os.path.join(root, file))

    return all_jsonl_files


def count_lines_jsonl(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)


def test():
    test_str = "/minimax-dialogue/data/users/beihai/M2_data/sources/pdf_hq/reward_filter/output/stem/EHQ/split_*/*/*.jsonl,/minimax-dialogue/data/users/beihai/M2_data/sources/pdf_lq/reward_filter/output/stem/EHQ/split_*/*/*.jsonl,/minimax-dialogue/data/users/beihai/M2_data/sources/institution_book/reward_filter/output/stem/EHQ/split_*/*/*.jsonl"
    res = get_all_jsonl_files(test_str)
    for each in res:
        print(each)


def check_duplicate_list(input_list):
    temp_list = []
    for each in input_list:
        if each not in temp_list:
            temp_list.append(each)
        if each in input_list:
            # print("Duplicate:", each)
            return True
    return False


if __name__ == '__main__':
    test()

