import re

import pandas as pd
import matplotlib.pyplot as plt


def display_means(df, label):
    print(f"Mean Values for {label}:")
    print(df.mean())
    print("\n")


def plot_metrics(df_local, df_cloud, dataset):
    metrics = ['rougeL', 'BERTScore F1', 'time_taken']
    # Visualization of comparison
    # for metric in metrics:
    #     plt.figure()
    #     plt.bar(['Cloud', 'Local'], [df_cloud[metric].mean(), df_local[metric].mean()])
    #     plt.title(f"Comparison of {metric}")
    #     plt.ylabel(metric)
    #     plt.xlabel('Environment')
    #     plt.show()
    #
    # Distribution of metrics
    for metric in metrics:
        plt.figure()
        plt.hist(df_cloud[metric], bins=30, alpha=0.5, label='Cloud')
        plt.hist(df_local[metric], bins=30, alpha=0.5, label='Local')
        plt.title(f"Distribution of {metric} ({dataset}")
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f"./plots/distribution_of_{metric}_for_{dataset}.png")

    # Scatter plots for quality metrics vs length of text
    quality_metrics = ['rougeL', 'BERTScore F1']

    for metric in quality_metrics:
        plt.figure(figsize=(10, 6))
        plt.scatter(df_cloud['len(text)'], df_cloud[metric], alpha=0.5, label='Cloud', color='blue')
        plt.scatter(df_local['len(text)'], df_local[metric], alpha=0.5, label='Local', color='orange')
        plt.title(f"{metric} vs Length of Text ({dataset})")
        plt.xlabel('Length of Text')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f"./plots/{metric}_vs_text_length_for_{dataset}.png")

    # Scatter plots for quality metrics vs time taken
    # for metric in quality_metrics:
    #     plt.figure(figsize=(10, 6))
    #     plt.scatter(df_cloud['time_taken'], df_cloud[metric], alpha=0.5, label='Cloud', color='blue')
    #     plt.scatter(df_local['time_taken'], df_local[metric], alpha=0.5, label='Local', color='orange')
    #     plt.title(f"{metric} vs Time Taken")
    #     plt.xlabel('Time Taken (s)')
    #     plt.ylabel(metric)
    #     plt.legend()
    #     plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(df_cloud['len(text)'], df_cloud['time_taken'], alpha=0.5, label='Cloud', color='blue')
    plt.scatter(df_local['len(text)'], df_local['time_taken'], alpha=0.5, label='Local', color='orange')
    plt.title(f"Text Length vs Time Taken ({dataset}")
    plt.xlabel('Length of Text')
    plt.ylabel('Time Taken (s)')
    plt.legend()
    plt.savefig(f"./plots/time_taken_vs_text_length_for_{dataset}.png")


def metric_comparison(df_local, df_cloud):
    # Metric comparison between Cloud and Local
    metrics = ['rougeL', 'BERTScore F1', 'time_taken']
    comparison = pd.DataFrame({
        'Metric': metrics,
        'Cloud Mean': [df_cloud[metric].mean() for metric in metrics],
        'Local Mean': [df_local[metric].mean() for metric in metrics]
    })
    print(comparison)


def extract_time(time_str):
    return float(re.search(r"[\d.]+", time_str).group())


if __name__ == "__main__":
    print("-----------X-Sum----------")
    df_local = pd.read_csv('evaluation_results_local_x_sum.csv')
    df_local['time_taken'] = df_local['time_taken'].apply(extract_time)
    df_local['memory_used'] = df_local['memory_used'].apply(extract_time)
    df_cloud = pd.read_csv('evaluation_results_cloud_x_sum.csv')
    df_cloud['time_taken'] = df_cloud['time_taken'].apply(extract_time)
    local_columns = ['len(text)', 'len(summary)', 'len(target_summary)', 'time_taken', 'memory_used', 'rougeL',
                     'BERTScore F1']
    cloud_columns = ['len(text)', 'len(summary)', 'len(target_summary)', 'time_taken', 'rougeL', 'BERTScore F1']
    display_means(df_local[local_columns], 'Local')
    display_means(df_cloud[cloud_columns], 'Cloud')

    metric_comparison(df_local, df_cloud)
    plot_metrics(df_local, df_cloud, 'X-sum')

    print("-----------Arxiv----------")
    df_local = pd.read_csv('evaluation_results_local_arxiv.csv')
    df_local['time_taken'] = df_local['time_taken'].apply(extract_time)
    df_local['memory_used'] = df_local['memory_used'].apply(extract_time)
    df_cloud = pd.read_csv('evaluation_results_cloud_arxiv.csv')
    df_cloud['time_taken'] = df_cloud['time_taken'].apply(extract_time)
    local_columns = ['len(text)', 'len(summary)', 'len(target_summary)', 'time_taken', 'memory_used', 'rougeL',
                     'BERTScore F1']
    cloud_columns = ['len(text)', 'len(summary)', 'len(target_summary)', 'time_taken', 'rougeL', 'BERTScore F1']
    display_means(df_local[local_columns], 'Local')
    display_means(df_cloud[cloud_columns], 'Cloud')

    metric_comparison(df_local, df_cloud)
    plot_metrics(df_local, df_cloud, 'arxiv')

    print("-----------Gov-Report----------")
    df_local = pd.read_csv('evaluation_results_local_gov_report.csv')
    df_local['time_taken'] = df_local['time_taken'].apply(extract_time)
    df_local['memory_used'] = df_local['memory_used'].apply(extract_time)
    df_cloud = pd.read_csv('evaluation_results_cloud_gov_report.csv')
    df_cloud['time_taken'] = df_cloud['time_taken'].apply(extract_time)
    local_columns = ['len(text)', 'len(summary)', 'len(target_summary)', 'time_taken', 'memory_used', 'rougeL',
                     'BERTScore F1']
    cloud_columns = ['len(text)', 'len(summary)', 'len(target_summary)', 'time_taken', 'rougeL', 'BERTScore F1']
    display_means(df_local[local_columns], 'Local')
    display_means(df_cloud[cloud_columns], 'Cloud')

    metric_comparison(df_local, df_cloud)
    plot_metrics(df_local, df_cloud, 'gov-report')


