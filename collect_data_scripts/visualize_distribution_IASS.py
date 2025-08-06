import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats


def main():
    with open("../local_data/test_data_0731/qwen3_sample_100_each_data_with_IASS.json", "r") as fi:
        data = json.load(fi)
    # print(data[0]["agent_cpt_dict"]["IASS_Score"])
    sta_data = []
    dimension_names = []
    display_sample = []
    for each in data:
        if "reward_dict_transformed" not in each:
            continue
        score = each["agent_cpt_dict"]["IASS_Score"]
        if score is None:
            continue
        curr_dict = {}
        iass_total = 0
        for k, v in score.items():
            if v["score"] >= 4 and each not in display_sample:
                display_sample.append(each)
            k = k.replace("goal-orientedness", "goal_orientedness")
            if "risk_assessment" in k:
                k = "risk_assessment_and_trade_offs"
            if k not in dimension_names:
                dimension_names.append(k)
            if k not in curr_dict:
                curr_dict[k] = v["score"]
            iass_total += v["score"]
        curr_dict["IASS_Total"] = iass_total
        if "IASS_Total" not in dimension_names:
            dimension_names.append("IASS_Total")
        score = each["reward_dict_transformed"]
        if score is None:
            continue
        for k, v in score.items():
            # if not k.startswith("score-reasoning"):
            #     continue
            if not k.startswith("score-"):
                continue
            if v >= 4 and each not in display_sample:
                display_sample.append(each)
            if k not in dimension_names:
                dimension_names.append(k)
            if k not in curr_dict:
                curr_dict[k] = v

        sta_data.append(curr_dict)
    print("dimension_names", dimension_names)
    print("len(sta_data)", len(sta_data))
    analyze_dimensions(sta_data, dimension_names)
    with open("../local_data/test_data_0731/display_sample_0805.json", "w") as fo:
        fo.write(json.dumps(display_sample, indent=4))


def analyze_dimensions(data_list, dimension_names, output_dir='output_figures_0805'):
    """
    分析多个维度指标的分布情况并可视化

    参数:
    data_list: 列表，每个元素是一个字典，包含各维度的评分
    dimension_names: 维度名称列表
    output_dir: 输出图片的目录

    返回:
    stats_df: 包含各维度统计信息的DataFrame
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 将列表转换为DataFrame
    df = pd.DataFrame(data_list)

    # 确保所有需要的维度都在DataFrame中
    missing_dims = [dim for dim in dimension_names if dim not in df.columns]
    if missing_dims:
        print(f"警告: 以下维度在数据中未找到: {missing_dims}")

    # 只保留指定的维度
    available_dims = [dim for dim in dimension_names if dim in df.columns]
    df = df[available_dims]

    # 创建统计信息DataFrame
    stats_df = pd.DataFrame(index=available_dims)

    # 计算各个统计量
    stats_df['样本数'] = df.count()
    stats_df['平均值'] = df.mean()
    stats_df['中位数'] = df.median()
    stats_df['标准差'] = df.std()
    stats_df['最小值'] = df.min()
    stats_df['最大值'] = df.max()
    stats_df['90%分位数'] = df.quantile(0.90)
    stats_df['95%分位数'] = df.quantile(0.95)

    # 打印统计信息
    print("各维度分布情况统计:")
    print(stats_df)

    # 1. 统计各维度每个评分(1-5分)的数量，并创建条形图
    # 创建一个新的DataFrame来存储每个分数的计数
    score_counts = pd.DataFrame(index=range(1, 6), columns=available_dims)

    for dim in available_dims:
        # 统计每个分数的数量
        counts = df[dim].value_counts().sort_index()
        for score in range(1, 6):
            score_counts.loc[score, dim] = counts.get(score, 0)

    # 绘制每个维度的评分分布条形图
    n_dims = len(available_dims)
    print("n_dims", n_dims)

    n_cols = 6  # 每行3个图
    n_rows = (n_dims + n_cols - 1) // n_cols  # 计算需要的行数
    print("n_cols", n_cols)
    print("n_rows", n_rows)
    input("enter")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 24))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])  # 确保axes是数组
    axes = axes.flatten()

    for i, dim in enumerate(available_dims):
        # 绘制条形图
        scores = score_counts[dim].index
        counts = score_counts[dim].values
        bars = axes[i].bar(scores, counts, color='skyblue', edgecolor='black')

        # 在每个条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2., height + 5,
                         f'{int(height)}', ha='center', va='bottom')

        axes[i].set_title(f'{dim}评分分布')
        axes[i].set_xlabel('评分')
        axes[i].set_ylabel('数量')
        axes[i].set_xticks(range(1, 6))

        # 添加均值标注
        mean_val = df[dim].mean()
        axes[i].axvline(mean_val, color='r', linestyle='--')
        axes[i].text(mean_val + 0.1, axes[i].get_ylim()[1] * 0.9, f'均值: {mean_val:.2f}',
                     color='red')

    # 隐藏多余的子图
    # for i in range(n_dims, len(axes)):
    #     axes[i].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, '各维度评分分布.png'), dpi=300, bbox_inches='tight')

    # 2. 各个维度指标两两的相关性
    # 计算相关系数矩阵
    corr = df[available_dims].corr()

    # 绘制相关性热图
    plt.figure(figsize=(18, 16))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # 创建上三角掩码

    # 使用seaborn绘制热图
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, fmt='.2f')
    plt.title('维度间相关性热图', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '维度间相关性热图.png'), dpi=300, bbox_inches='tight')

    # 创建一个聚合的条形图，显示所有维度的评分分布比较
    plt.figure(figsize=(40, 10))

    # 转换数据为绘图格式
    plot_data = []
    for dim in available_dims:
        for score in range(1, 6):
            plot_data.append({
                '维度': dim,
                '评分': score,
                '数量': score_counts.loc[score, dim]
            })

    plot_df = pd.DataFrame(plot_data)

    # 绘制聚合条形图
    ax = sns.barplot(x='维度', y='数量', hue='评分', data=plot_df, palette='viridis')
    plt.title('所有维度评分分布对比', fontsize=16)
    plt.xlabel('维度', fontsize=14)
    plt.ylabel('样本数量', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='评分')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '所有维度评分分布对比.png'), dpi=300, bbox_inches='tight')

    # 保存原始数据表格
    stats_df.to_csv(os.path.join(output_dir, '维度统计结果.csv'))
    score_counts.to_csv(os.path.join(output_dir, '维度评分计数.csv'))

    print(f"所有图表和数据已保存到 {output_dir} 目录")

    # 显示图表
    # plt.show()

    return stats_df, score_counts


if __name__ == "__main__":
    main()


