import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, friedmanchisquare, wilcoxon
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats import multicomp
import statsmodels.api as sm

def run_stat_tests(data_phase_1, data_phase_2, data_phase_3):
    # Ensure all data are numpy arrays
    data_phase_1 = np.array(data_phase_1)
    data_phase_2 = np.array(data_phase_2)
    data_phase_3 = np.array(data_phase_3)

    # Make all phases the same length by truncating to the length of the smallest phase
    min_len = min(len(data_phase_1), len(data_phase_2), len(data_phase_3))

    # Truncate the data
    data_phase_1 = data_phase_1[:min_len]
    data_phase_2 = data_phase_2[:min_len]
    data_phase_3 = data_phase_3[:min_len]

    # Print the number of data points used
    print(f"Number of data points per phase after truncation: {min_len}")

    # Shapiro-Wilk normality test
    print("\nNormality Test (Shapiro-Wilk):")
    for phase_data, phase_name in zip(
        [data_phase_1, data_phase_2, data_phase_3], 
        ['Phase 1', 'Phase 2', 'Phase 3']
    ):
        stat, p = shapiro(phase_data)
        print(f"{phase_name}: stat={stat:.6e}, p-value={p:.6e}")

    # Repeated Measures ANOVA (if data is normally distributed)
    # Uncomment the following code if normality assumption is met
    """
    print("\nRunning Repeated Measures ANOVA...")
    try:
        data_long = pd.DataFrame({
            'Subject': np.tile(np.arange(min_len), 3),
            'Phase': np.repeat(['Phase 1', 'Phase 2', 'Phase 3'], min_len),
            'Diameter': np.concatenate([data_phase_1, data_phase_2, data_phase_3])
        })
        anova_results = AnovaRM(data_long, 'Diameter', 'Subject', within=['Phase']).fit()
        print(anova_results.summary())
    except Exception as e:
        print(f"Error with Repeated Measures ANOVA: {e}")
    """

    # Friedman's Test (non-parametric alternative)
    print("\nRunning Friedman's Test...")
    try:
        friedman_test = friedmanchisquare(data_phase_1, data_phase_2, data_phase_3)
        print(f"Friedman test statistic={friedman_test.statistic:.6e}, p-value={friedman_test.pvalue:.6e}")
    except Exception as e:
        print(f"Error with Friedman's test: {e}")

    # Wilcoxon Signed-Rank Tests
    print("\nWilcoxon Signed-Rank Tests:")
    comparisons = [
        ('Phase 1', 'Phase 2', data_phase_1, data_phase_2),
        ('Phase 2', 'Phase 3', data_phase_2, data_phase_3),
        ('Phase 1', 'Phase 3', data_phase_1, data_phase_3)
    ]
    for phase_a, phase_b, data_a, data_b in comparisons:
        stat, p = wilcoxon(data_a, data_b)
        print(f"{phase_a} vs {phase_b}: stat={stat:.6e}, p-value={p:.6e}")


def main():
    
    # Load the CSV file
    csv_file_path = '/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/Participant_10/2023-02-10 10:23:22_anatT_baseline_P10T1/000/extracted_diameter.csv'
    df = pd.read_csv(csv_file_path)

    # Get the value of the timestamp in the fifth row
    fifth_row_timestamp = df['timestamp'].iloc[4]

    # Check if any of the first four rows have a timestamp that differs from the fifth row by more than 1000
    rows_to_delete = df['timestamp'].iloc[:4].apply(lambda x: abs(x - fifth_row_timestamp) > 1000)

    # Delete those rows
    df = df.drop(df.index[:4][rows_to_delete])

    # Ensure the 'topic' column is treated as strings and handle NaN values
    df['topic'] = df['topic'].astype(str)

    # Treat the first timestamp as t=0 seconds and calculate time in seconds
    start_time = df['timestamp'].min()
    df['time_seconds'] = df['timestamp'] - start_time

    # Remove rows in which both diameters are 0
    df = df[~((df['diameter_2d [px]'] == 0) & (df['diameter_3d [mm]'] == 0))]

    # Create dataframes of just 2d or just 3d
    df_2d = df[~df['topic'].str.contains('3d')]
    df_2d = df_2d[~((df['diameter_2d [px]'] == 0))]
    df_3d = df[~df['topic'].str.contains('2d')]
    df_3d = df_3d[~((df['diameter_3d [mm]'] == 0))]

   # Filter data for the right and left eyes
    right_eye_2d = df_2d[df_2d['eye_id'] == 0.0]
    left_eye_2d = df_2d[df_2d['eye_id'] == 1.0]
    right_eye_3d = df_3d[df_3d['eye_id'] == 0.0]
    left_eye_3d = df_3d[df_3d['eye_id'] == 1.0]

    # Calculate aggregated avg for both eyes in 2d and 3d
    both_eyes_2d = df_2d.groupby('time_seconds')['diameter_2d [px]'].mean().reset_index()
    both_eyes_3d = df_3d.groupby('time_seconds')['diameter_3d [mm]'].mean().reset_index()

    # Combined 2D Graph (with separate axes for Pupil 0 and Pupil 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6))
    
    # Pupil 0 - 2D Diameter
    ax1.plot(right_eye_2d['time_seconds'], right_eye_2d['diameter_2d [px]'], color='red', linewidth=0.6)
    ax1.set_title('Right Pupil Diameter Over Time (2D, pixels)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Diameter (px)')
    ax1.grid(True)

    # Pupil 1 - 2D Diameter
    ax2.plot(left_eye_2d['time_seconds'], left_eye_2d['diameter_2d [px]'], color='blue', linewidth=0.6)
    ax2.set_title('Left Pupil Diameter Over Time (2D, pixels)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Diameter (px)')
    ax2.grid(True)

    # Save the 2D graph (by eye)
    output_path_2d = '/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/Participant_10/2023-02-10 10:23:22_anatT_baseline_P10T1/000/pupil_diameter_2d_graph.jpg'
    plt.tight_layout()
    plt.savefig(output_path_2d, format='jpg')

     # Combined 3D Graph (with separate axes for Pupil 0 and Pupil 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6))
    
    # Pupil 0 - 3D Diameter
    ax1.plot(right_eye_3d['time_seconds'], right_eye_3d['diameter_3d [mm]'], color='red', linewidth=0.6)
    ax1.set_title('Right Pupil Diameter Over Time (3D, mm)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Diameter (mm)')
    ax1.grid(True)

    # Pupil 1 - 3D Diameter
    ax2.plot(left_eye_3d['time_seconds'], left_eye_3d['diameter_3d [mm]'], color='blue', linewidth=0.6)
    ax2.set_title('Left Pupil Diameter Over Time (3D, mm)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Diameter (mm)')
    ax2.grid(True)

    # Save the 3D graph (by eye)
    output_path_3d = '/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/Participant_10/2023-02-10 10:23:22_anatT_baseline_P10T1/000/pupil_diameter_3d_graph.jpg'
    plt.tight_layout()
    plt.savefig(output_path_3d, format='jpg')

    # Combined 2D/3D Graph
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6))

    # Both Pupils - 2D Diameter
    ax1.plot(both_eyes_2d['time_seconds'], both_eyes_2d['diameter_2d [px]'], color='blue', linewidth=0.6)
    ax1.set_title('Average Pupil Diameter Over Time (2D, pixels)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Average Diameter (px)')
    ax1.grid(True)

    # Both Pupils - 3D Diameter
    ax2.plot(both_eyes_3d['time_seconds'], both_eyes_3d['diameter_3d [mm]'], color='purple', linewidth=0.6)
    ax2.set_title('Average Pupil Diameter Over Time (3D, mm)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Average Diameter (mm)')
    ax2.grid(True)

    # Save the averaged 2D graph
    output_path_avg_2d3d = '/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/Participant_10/2023-02-10 10:23:22_anatT_baseline_P10T1/000/pupil_diameter_2d3d_average_graph.jpg'
    plt.tight_layout()
    plt.savefig(output_path_avg_2d3d, format='jpg')

    # Define the time ranges for the three phases
    phase_1_2d = (both_eyes_2d['time_seconds'] <= 150)
    phase_2_2d = (both_eyes_2d['time_seconds'] > 150) & (both_eyes_2d['time_seconds'] <= 188)
    phase_3_2d = (both_eyes_2d['time_seconds'] > 188)

    phase_1_3d = (both_eyes_3d['time_seconds'] <= 150)
    phase_2_3d = (both_eyes_3d['time_seconds'] > 150) & (both_eyes_3d['time_seconds'] <= 188)
    phase_3_3d = (both_eyes_3d['time_seconds'] > 188)

    # Prepare box plot data
    data_2d_phase_1 = both_eyes_2d[phase_1_2d]['diameter_2d [px]']
    data_2d_phase_2 = both_eyes_2d[phase_2_2d]['diameter_2d [px]']
    data_2d_phase_3 = both_eyes_2d[phase_3_2d]['diameter_2d [px]']
    
    data_3d_phase_1 = both_eyes_3d[phase_1_3d]['diameter_3d [mm]']
    data_3d_phase_2 = both_eyes_3d[phase_2_3d]['diameter_3d [mm]']
    data_3d_phase_3 = both_eyes_3d[phase_3_3d]['diameter_3d [mm]']

    # Create the box plots for the three phases (2D)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([data_2d_phase_1, data_2d_phase_2, data_2d_phase_3],
               labels=['Phase 1', 'Phase 2', 'Phase 3'])
    
    ax.set_title('Box Plot of Pupil 2D Size Across Phases')
    ax.set_ylabel('2D Diameter (px)')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Remove gridlines
    ax.grid(False)

    # Save the box plot
    output_path_box = '/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/Participant_10/2023-02-10 10:23:22_anatT_baseline_P10T1/000/pupil_diameter_2d_boxplot_by_phase.jpg'
    plt.tight_layout()
    plt.savefig(output_path_box, format='jpg')

    # Create the box plots for the three phases (3D)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([data_3d_phase_1, data_3d_phase_2, data_3d_phase_3],
               labels=['Phase 1', 'Phase 2', 'Phase 3'])
    
    ax.set_title('Box Plot of Pupil 3D Size Across Phases')
    ax.set_ylabel('3D Diameter (mm)')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Remove gridlines
    ax.grid(False)

    # Save the box plot
    output_path_box = '/Users/orenw/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/Participant_10/2023-02-10 10:23:22_anatT_baseline_P10T1/000/pupil_diameter_3d_boxplot_by_phase.jpg'
    plt.tight_layout()
    plt.savefig(output_path_box, format='jpg')


    # Run the statistical tests (for either 2D or 3D)
    run_stat_tests(data_3d_phase_1, data_3d_phase_2, data_3d_phase_3)

    # Return the output paths
    return output_path_2d, output_path_3d, output_path_box
    

# Run the function
main()
