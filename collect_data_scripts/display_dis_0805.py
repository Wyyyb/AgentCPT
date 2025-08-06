import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    with open("../local_data/test_data_0731/qwen3_sample_100_each_data_with_IASS.json", "r") as fi:
        data = json.load(fi)
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
        curr_dict["IASS_Total"] = iass_total / 12.0
        if "IASS_Total" not in dimension_names:
            dimension_names.append("IASS_Total")
        score = each["reward_dict_transformed"]
        if score is None:
            continue
        for k, v in score.items():
            if not k.startswith("score-"):
                continue
            # if v >= 4 and each not in display_sample:
            #     display_sample.append(each)
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


def score_to_interval(score):
    """
    将分数转换到对应的区间
    0-1: [0,1)
    1-2: [1,2)
    2-3: [2,3)
    3-4: [3,4)
    4-5: [4,5]
    """
    if pd.isna(score):
        return None
    if score < 1:
        return '[0,1)'
    elif score < 2:
        return '[1,2)'
    elif score < 3:
        return '[2,3)'
    elif score < 4:
        return '[3,4)'
    else:  # score >= 4
        return '[4,5]'


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

    # 创建子目录用于存放单独的条形图
    bar_charts_dir = os.path.join(output_dir, 'individual_bar_charts')
    os.makedirs(bar_charts_dir, exist_ok=True)

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

    # 1. 统计各维度在不同区间的数量
    interval_labels = ['[0,1)', '[1,2)', '[2,3)', '[3,4)', '[4,5]']
    score_counts = pd.DataFrame(index=interval_labels, columns=available_dims)

    for dim in available_dims:
        # 将分数转换为区间
        intervals = df[dim].apply(score_to_interval)
        # 统计每个区间的数量
        interval_counts = intervals.value_counts()

        for interval in interval_labels:
            score_counts.loc[interval, dim] = interval_counts.get(interval, 0)

    # 为每个维度单独创建条形图
    for dim in available_dims:
        plt.figure(figsize=(12, 6))

        # 获取该维度的评分分布
        intervals = score_counts[dim].index
        counts = score_counts[dim].values

        # 创建条形图，使用不同颜色表示不同区间
        colors = ['#ff9999', '#ffcc99', '#ffff99', '#99ff99', '#99ccff']
        bars = plt.bar(range(len(intervals)), counts, color=colors, edgecolor='black', alpha=0.7)

        # 在每个条形上方添加数值标签
        max_count = max(counts) if max(counts) > 0 else 1
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:  # 只在有数据时添加标签
                plt.text(bar.get_x() + bar.get_width() / 2., height + max_count * 0.02,
                         f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 设置图表属性
        plt.title(f'{dim} 评分分布', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('评分区间', fontsize=14)
        plt.ylabel('样本数量', fontsize=14)
        plt.xticks(range(len(intervals)), intervals, fontsize=11)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')

        # 添加均值标注
        mean_val = df[dim].mean()
        plt.text(0.02, 0.85, f'均值: {mean_val:.2f}', transform=plt.gca().transAxes,
                 color='red', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # 添加样本数量和数据范围信息
        total_samples = df[dim].count()
        min_val = df[dim].min()
        max_val = df[dim].max()
        info_text = f'总样本数: {total_samples}\n数据范围: [{min_val:.2f}, {max_val:.2f}]'
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        # 设置y轴范围，留出空间给标签
        plt.ylim(0, max_count * 1.15)

        plt.tight_layout()

        # 保存图片，文件名中的特殊字符替换为下划线
        safe_filename = dim.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?',
                                                                                                             '_').replace(
            '"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        plt.savefig(os.path.join(bar_charts_dir, f'{safe_filename}_评分分布.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # 关闭图形以释放内存

        print(f"已保存: {dim} 的条形图")

    # 打印区间统计详情
    print("\n各维度区间分布统计:")
    print(score_counts)

    # 2. 各个维度指标两两的相关性
    corr = df[available_dims].corr()

    # 绘制相关性热图
    plt.figure(figsize=(18, 16))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # 创建上三角掩码

    # 使用seaborn绘制热图
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, fmt='.2f', square=True)
    plt.title('维度间相关性热图', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '维度间相关性热图.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 创建一个聚合的条形图，显示所有维度的评分分布比较
    plt.figure(figsize=(max(len(available_dims) * 2, 20), 10))

    # 转换数据为绘图格式
    plot_data = []
    for dim in available_dims:
        for interval in interval_labels:
            plot_data.append({
                '维度': dim,
                '评分区间': interval,
                '数量': score_counts.loc[interval, dim]
            })

    plot_df = pd.DataFrame(plot_data)

    # 绘制聚合条形图
    ax = sns.barplot(x='维度', y='数量', hue='评分区间', data=plot_df,
                     palette=['#ff9999', '#ffcc99', '#ffff99', '#99ff99', '#99ccff'])
    plt.title('所有维度评分分布对比', fontsize=16)
    plt.xlabel('维度', fontsize=14)
    plt.ylabel('样本数量', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='评分区间', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '所有维度评分分布对比.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存原始数据表格
    stats_df.to_csv(os.path.join(output_dir, '维度统计结果.csv'), encoding='utf-8-sig')
    score_counts.to_csv(os.path.join(output_dir, '维度区间计数.csv'), encoding='utf-8-sig')

    print(f"所有图表和数据已保存到 {output_dir} 目录")
    print(f"各维度单独条形图已保存到 {bar_charts_dir} 目录")

    return stats_df, score_counts


if __name__ == "__main__":
    main()