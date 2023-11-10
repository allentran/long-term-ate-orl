import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    df = pd.read_csv('qw_vs_surrogates.csv')

    plt.style.use('seaborn-pastel')

    # Assuming T is your index
    T_values = df['T'].replace(np.inf, 100).unique()

    # Initialize lists to store mean and confidence interval values
    mean_qw = []
    mean_naive = []
    mean_surrogate_low = []
    mean_surrogate_mid = []
    mean_surrogate_all = []
    mean_ate = []
    lower_qw = []
    upper_qw = []
    lower_naive = []
    upper_naive = []
    lower_surrogate_low = []
    upper_surrogate_low = []
    lower_surrogate_mid = []
    upper_surrogate_mid = []
    lower_surrogate_all = []
    upper_surrogate_all = []

    # Calculate the mean and confidence intervals for each T
    for T in T_values:
        subset_surrogates = df.loc[df['T'].replace(np.inf, 100) == T]
        subset = df.loc[df['T'].replace(np.inf, 100) == T]
        mean_qw.append(np.mean(subset['ate_qw']))
        mean_ate.append(np.mean(subset['ate']))
        mean_naive.append(np.mean(subset['ate_naive']))
        mean_surrogate_low.append(np.mean(subset_surrogates['ate_surrogate']))
        mean_surrogate_mid.append(np.mean(subset_surrogates['ate_mid_both_surrogate']))
        mean_surrogate_all.append(np.mean(subset_surrogates['ate_all_both_surrogate']))
        lower_qw.append(subset['ate_qw'].quantile(0.05))
        upper_qw.append(subset['ate_qw'].quantile(0.95))
        lower_naive.append(subset['ate_naive'].quantile(0.05))
        upper_naive.append(subset['ate_naive'].quantile(0.95))
        lower_surrogate_low.append(subset_surrogates['ate_surrogate'].quantile(0.05))
        upper_surrogate_low.append(subset_surrogates['ate_surrogate'].quantile(0.95))
        lower_surrogate_mid.append(subset_surrogates['ate_mid_both_surrogate'].quantile(0.05))
        upper_surrogate_mid.append(subset_surrogates['ate_mid_both_surrogate'].quantile(0.95))
        lower_surrogate_all.append(subset_surrogates['ate_all_both_surrogate'].quantile(0.05))
        upper_surrogate_all.append(subset_surrogates['ate_all_both_surrogate'].quantile(0.95))

    # Create a new figure
    plt.figure()

    # Plot the mean lines with color variables and different markers
    orl_line = plt.plot(T_values, mean_qw, label='ORL', marker='D')
    surrogate_line = plt.plot(T_values, mean_surrogate_low, marker='s', label='Surrogate (Short XP)')
    ate_line = plt.plot(T_values, mean_ate, label='ATE')

    surrogate_mid_line = plt.plot(T_values, mean_surrogate_mid, marker='^', label='Surrogate (Mid)')
    surrogate_all_line = plt.plot(T_values, mean_surrogate_all, marker='>', label='Surrogate (Data up to T)')
    naive_line = plt.plot(T_values, mean_naive, marker='<', label='Naive')

    color_qw = orl_line[0].get_color()
    color_ate = ate_line[0].get_color()
    color_naive = naive_line[0].get_color()
    color_surrogate = surrogate_line[0].get_color()
    color_mid_surrogate = surrogate_mid_line[0].get_color()
    color_all_surrogate = surrogate_all_line[0].get_color()

    # Plot the confidence intervals with corresponding color
    plt.fill_between(T_values, lower_qw, upper_qw, color=color_qw, alpha=.2)
    # plt.fill_between(T_values, lower_naive, upper_naive, color=color_naive, alpha=.2)
    plt.fill_between(T_values, lower_surrogate_low, upper_surrogate_low, color=color_surrogate, alpha=.2)
    # plt.fill_between(T_values, lower_surrogate_mid, upper_surrogate_mid, color=color_mid_surrogate, alpha=.2)
    plt.fill_between(T_values, lower_surrogate_all, upper_surrogate_all, color=color_all_surrogate, alpha=.2)

    # Add labels and title
    plt.xlabel('Duration of Treatment')
    plt.ylabel('Estimate')
    plt.legend(frameon=False, loc='center right', ncol=1)

    # Draw the plot and get labels
    plt.draw()
    ticks = plt.gca().get_xticks().tolist()

    # Add new tick if it doesn't exist
    if 1 not in ticks:
        ticks.append(1)
        ticks.sort()

    # Set the new ticks
    plt.gca().set_xticks(T_values)
    labels = [item.get_text() for item in plt.gca().get_xticklabels()]
    print(labels)
    # Change the last label
    # labels[-1] = ''
    labels[0] = '$\infty$'
    print(labels)

    # Set the new labels
    plt.gca().set_xticklabels(labels)
    plt.xlabel('Duration of treatment', fontsize=12)
    plt.ylabel('Estimate of long-term outcome', fontsize=12)

    # Set the font size for the tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()