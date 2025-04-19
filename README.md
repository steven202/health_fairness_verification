# Fairness Verification of Machine Learning Models in Alzheimer’s Disease Progression Prediction

## Environment Installation
To set up the environment, run the following command:

```bash
conda env create -f environment.yml
```

## Discrete Model Training & Fairness Verification

### Folder Navigation
First, navigate to the `discrete_train_verify` folder:

```bash
cd discrete_train_verify
```

### Data Location
Place the ADNI dataset in the following location: 

`fairness_data/TADPOLE_D1_D2.csv`.

### Training and Verification
To train, test, and verify the model, execute:

```bash
bash run_all.sh
```
*Note: Detailed commands are available in the `run_all.sh` file.*

- Results will be saved in the `generated_files` folder.
- Logs will be saved in the `logs` folder.

### Case Comparison
To compare the following four cases:
1. 3 years without repeated visits
2. 3 years with repeated visits
3. 5 years without repeated visits
4. 5 years with repeated visits

Run the following command for each case:

```bash
python extract_all_pnr_fpr.py
```
- Results will be saved in the `log_fnr_fpr` folder.

Once you have the results for all four cases (each with a `log_fnr_fpr` folder), open and run the `extract_acc_fnr_fpr_for_all_folders.ipynb` notebook in Jupyter. The comparison results will be saved to:

- `whole_df.csv`
- `plots_for_paper/discrete_fnr_fpr_acc.pdf`

### Clean Up
To clean up the running results, execute:

```bash
bash clean.sh
```
- This will remove unnecessary outputs but keep the `generated_files` and `logs` folders.

## Continuous Model Training

### Folder Navigation
First, navigate to the `continuous_train` folder:

```bash
cd continuous_train
```

### Data Location
Place the ADNI dataset in the following location: 

`fairness_data/TADPOLE_D1_D2.csv`.

### Training
To train and test the model, execute:

```bash
bash run.sh
```
- Results will be saved in the `checkpoints_03062024` folder.

## Continuous Model Fairness Verification

### Folder Navigation
First, navigate to the `continuous_verify/complete_verifier` folder:

```bash
cd continuous_verify/complete_verifier
```

### Data Location
Place the ADNI dataset in the following location: 

`fairness_data/TADPOLE_D1_D2.csv`.

### Verification
To verify the fairness of the continuous model, execute:

```bash
bash run.sh
```
- This script will run the verifier on the continuous model for all possible combinations of attributes with different perturbation $\epsilon_k$ values ranging from 1 to 15 to maximum values of the attribute.
- Results will be saved in the `output_year_eps_07042024` folder, and logs will be saved in the `log_07042024` folder.
- It may take a long time to run the script; use `bash monitor_progress.sh` to monitor the progress.

### Result Analysis
To analyze the results, run the following command:

```bash
bash run_analysis.sh
```
- This script will generate results in the `output_year_eps_07042024` folder, and logs will be saved in the `log_07042024` folder as well.
- The script `write_vnnlib_plot.ipynb` will generate the plots and data for the table in the paper.

## Trained Model Weights
The model weights can be accessed from the following link: [Trained Model Weights](https://drive.google.com/drive/folders/14qSE4P0sQ4ZxfH84V7ZjTkHUDnwOL22R?usp=sharing).

The repository includes three folders: `continuous_train`, `continuous_verify`, and `discrete_train_verify`. 

- For the `continuous_train` folder, model weights are saved in the `checkpoints_03062024` folder.
- For the `continuous_verify` folder, there are no model weights; instead, figures are in `images_cont` and logs are in `log_07042024`.
- For the `discrete_train_verify` folder, there are four subfolders corresponding to four cases:
  - `3_years_no_repeated_visits` (health_fairness_3Y)
  - `3_years_repeated_visits` (health_fairness_3Y_repeat)
  - `5_years_no_repeated_visits` (health_fairness_5Y_)
  - `5_years_repeated_visits` (health_fairness_5Y_repeat)

Inside each subfolder, model weights are saved in the `ckpt2` folder, along with other generated files.

Though the saved model weights are aimed to reproduce the results, due to the development cycle spanning over a year, there may be some discrepancies in the results, e.g., due to library version updates and reorganization of the code.

## Related Works
This project has used or is modified based on the official codes of [Nguyen2020_RNNAD](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/predict_phenotypes/Nguyen2020_RNNAD) and [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN). We appreciate the authors for their contributions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
