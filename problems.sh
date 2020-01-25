UNAME=$(uname)

if [ "$UNAME" == "Linux" ] ; 
	then
	echo "linux"
. ./.cgpvenv/bin/activate
	elif [[ "$UNAME" == CYGWIN* || "$UNAME" == MINGW* ]] ;
	then
	echo "windows"
. ./.cgpvenv/Scripts/activate	
	else
	echo "not supported os"
fi

declare -a fit=('vaf') #'mae' 'mse' 'r2_score' 'vaf')
declare -a probs=('keijzer_1' 'keijzer_2' 'keijzer_3' 'keijzer_4' 'keijzer_5' 'keijzer_6' 'keijzer_7' 'keijzer_8' 'keijzer_9' 'keijzer_10'
'keijzer_11' 'keijzer_12' 'keijzer_13' 'keijzer_14' 'keijzer_15' 'korn_1' 'korn_2' 'korn_3' 'korn_4' 'korn_5' 'korn_6' 'korn_7' 'korn_8'
'korn_9' 'korn_10' 'korn_11' 'korn_12' 'korn_13' 'korn_14' 'korn_15' 'simple_1')

ps=20							# POPULATION_SIZE
mg=100							# MAX_GENERATIONS
O='+,-,*,/,Sin,Cos,Pow2,Sqrt'	# USEDDOPERATORS
mp=0.23							# MUTATION_PROBABILITY
mc=15							# MUTATION_COUNTER
RS2=23							# RANDOM_SEED
L1='INFO'						# loglevel
L2='XXL'						# number of files exported

co=2							# CONSTANT_SIZE
CMIN=-15  						# constant range min
CMAX=15   						# constant range max
ROWS=1
C=35 							# COLUMNS
la=13 							# ES_LAMBDA
EE='mae,mse,r2_score,vaf' 		# EVALUATORS

# problems for eval
for P in "${probs[@]}"
do
    # generations
    for F in "${fit[@]}"
    do
echo $P $F
python ./main.py -O $O -e $F -ee $EE -c $C -mg $mg -pm $mp -mc $mc -P $P -C $co $CMIN $CMAX -la $la -ps $ps -rs $RS2 -log $L1 $L2
echo "done"
    done
done

echo "all done"
