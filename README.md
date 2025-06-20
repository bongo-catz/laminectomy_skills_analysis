# Performance Metrics and Skills Analysis for Skull-Base Surgery
## CIS II, Spring 2024, Team 1
Team Members: Nimesh Nagururu, Hannah Puhov
Mentors: Dr. Adnan Munawar, Dr. Manish Sahu, Hisashi Ishida, Juan Barragan, Dr. Deepa Galaiya, Dr. Francis Creighton

## Main Files and Folders
All scripts are well commented, describing use and usage.
| Item | Path |
|-------|------------------|
| Read experiment from HDF5 files     | `~/scripts/exp_reader.py`          |
| Calculate performance metrics            | `~/scripts/metric_extractor.py`     |
| Visualize stroke metrics      | `~/scripts/plotting_skills_analysis.py`|
| Aggregate metrics across experiments for analysis    | `~/scripts/agg_metrics.py`   |
| Statistical comparison of novices and experts          | `~/output_SDF/` |

## Analysis Workflow

1. Run `metrics_extractor.py` to produce a directory containing different subdirectories containing stroke metrics from participants in user study
2. Run `merge_csv.py` to produce a merged_csv file containing all stroke metrics for ALL participants
3. Run `clustering.py` on merged_csv file