echo "start"

./eval_problem_korn_15_mae.sh &
./eval_problem_korn_15_mse.sh &
./eval_problem_korn_15_vaf.sh &
./eval_problem_korn_15_r2_score.sh &
wait
echo "finished"