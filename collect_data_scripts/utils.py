import os
import glob


def get_all_jsonl_files(path_string):
    """
    输入一个路径字符串，返回所有子jsonl文件路径的列表

    参数:
        path_string: 包含路径的字符串，可以是多个路径（用英文逗号分隔），支持通配符

    返回:
        所有jsonl文件的完整路径列表
    """
    all_jsonl_files = []

    # 按英文逗号分割路径
    paths = path_string.split(',')

    for path in paths:
        path = path.strip()  # 移除可能的空格

        # 如果路径直接以.jsonl结尾且包含通配符，直接使用glob获取匹配的文件
        if path.endswith('.jsonl') and ('*' in path or '?' in path):
            matching_files = glob.glob(path)
            all_jsonl_files.extend(matching_files)

        # 如果是目录路径（可能包含通配符）
        else:
            # 首先处理可能的通配符，获取所有匹配的目录
            if '*' in path or '?' in path:
                matching_dirs = glob.glob(path)
            else:
                matching_dirs = [path]

            # 对每个匹配的目录，递归查找所有.jsonl文件
            for directory in matching_dirs:
                if os.path.isdir(directory):
                    for root, _, files in os.walk(directory):
                        for file in files:
                            if file.endswith('.jsonl'):
                                all_jsonl_files.append(os.path.join(root, file))

    return all_jsonl_files


if __name__ == '__main__':
    test_str = "/minimax-dialogue/data/users/beihai/M2_data/sources/pdf_hq/reward_filter/output/stem/EHQ/split_*/*/*.jsonl,/minimax-dialogue/data/users/beihai/M2_data/sources/pdf_lq/reward_filter/output/stem/EHQ/split_*/*/*.jsonl,/minimax-dialogue/data/users/beihai/M2_data/sources/institution_book/reward_filter/output/stem/EHQ/split_*/*/*.jsonl"
    res = get_all_jsonl_files(test_str)
    for each in res:
        print(each)

