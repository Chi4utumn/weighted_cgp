echo "start"

./eval_problem_korn_14_mae.sh &
./eval_problem_korn_14_mse.sh &
./eval_problem_korn_14_vaf.sh &
./eval_problem_korn_14_r2_score.sh &
wait
echo "finished"