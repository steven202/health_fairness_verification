# Discrete Model Training & Fairness Verification

## Data Location
The ADNI dataset should be placed in the following location: 

`fairness_data/TADPOLE_D1_D2.csv`.

## Environment Installation
To set up the environment, run the following command:

```bash
conda env create -f environment.yml
```

## Training and Verification
To train and verify the model, execute:

```bash
bash run_all.sh
```
*Note: Detailed commands are available in the `run_all.sh` file.*

- The results will be saved in the `generated_files` folder.
- Logs will be saved in the `logs` folder.

## Case Comparison
To compare the following 4 cases:
1. 3 years without repeated visits
2. 3 years with repeated visits
3. 5 years without repeated visits
4. 5 years with repeated visits

Run the following command for each case:

```bash
python extract_all_pnr_fpr.py
```
- The results will be saved in the `log_fnr_fpr` folder.

Once you have the results for all 4 cases (each with a `log_fnr_fpr` folder), open and run the `extract_acc_fnr_fpr_for_all_folders.ipynb` notebook in Jupyter. The comparison results will be saved to:

- `whole_df.csv`
- `plots_for_paper/discrete_fnr_fpr_acc.pdf`

## Clean Up
To clean up the running results, execute:

```bash
bash clean.sh
```
- This will remove the unnecessary outputs but keep the `generated_files` and `logs` folders.

## Related Works
This project has used or is based on the official codes of [Nguyen2020_RNNAD](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/predict_phenotypes/Nguyen2020_RNNAD) and [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.