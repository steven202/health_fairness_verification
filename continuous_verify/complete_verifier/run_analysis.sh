#! /bin/bash
# Allow the user to break the script with Ctrl-C
trap "exit" INT

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate alpha-beta-crown

main_dir="output_year_eps_07042024"
log_dir="log_07042024"
for year_eps in -1 3 6 10 15 1 5 7 9 11 13 2 4 8 12 14
do 
    # parser.add_argument("--vnnlib-execute-dir", default="write_vnnlib_execute_dir")
    # parser.add_argument("--vnnlib-result-dir", default="write_vnnlib_verified_result_dir")
    python write_vnnlib_extract_6.py --vnnlib-execute-dir $main_dir/output_$year_eps/write_vnnlib_execute_dir --vnnlib-result-dir $main_dir/output_$year_eps/write_vnnlib_verified_result_dir > $log_dir/extract_$year_eps.txt
    python write_vnnlib_analysis_7.py --vnnlib-result-dir $main_dir/output_$year_eps/write_vnnlib_verified_result_dir > $log_dir/analysis_$year_eps.txt
    cp $log_dir/extract_$year_eps.txt $main_dir/output_$year_eps
    cp $log_dir/analysis_$year_eps.txt $main_dir/output_$year_eps
done

if [ -f $log_dir/analysis_age.txt ]; then
    rm $log_dir/analysis_age.txt
fi
if [ -f $log_dir/analysis_education.txt ]; then
    rm $log_dir/analysis_education.txt
fi
if [ -f $log_dir/analysis_age_education.txt ]; then
    rm $log_dir/analysis_age_education.txt
fi
echo "age analysis" > $log_dir/analysis_age.txt
echo "education analysis" > $log_dir/analysis_education.txt
echo "age education analysis" > $log_dir/analysis_age_education.txt
for year_eps in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 -1
do 
    echo "year_eps: $year_eps" >> $log_dir/analysis_age.txt
    echo "year_eps: $year_eps" >> $log_dir/analysis_education.txt
    echo "year_eps: $year_eps" >> $log_dir/analysis_age_education.txt
    cat $log_dir/analysis_$year_eps.txt | grep " combination AGE " >> $log_dir/analysis_age.txt
    cat $log_dir/analysis_$year_eps.txt | grep " combination PTEDUCAT " >> $log_dir/analysis_education.txt
    cat $log_dir/analysis_$year_eps.txt | grep " combination AGE_PTEDUCAT " >> $log_dir/analysis_age_education.txt
done
