UNAME=$(uname)

if [ "$UNAME" == "Linux" ] ; 
	then
	echo "linux"
. ../.cgpvenv/bin/activate
	elif [[ "$UNAME" == CYGWIN* || "$UNAME" == MINGW* ]] ;
	then
	echo "windows"
. ../.cgpvenv/Scripts/activate	
	else
	echo "not supported os"
fi

P="keijzer_1"                 	# PROBLEM

ps=332 	                		# POPULATION_SIZE
mg=111                       	# MAX_GENERATIONS
O='+,-,*,/,Sin,Cos,Pow2,Sqrt'	# USEDDOPERATORS
mp=0.16                        # MUTATION_PROBABILITY
mc=15                          # MUTATION_COUNTER
RS1=23                          # RANDOM_SEED
RS2=-1							# RANDOM RANDOM_SEED

co=0                           # CONSTANT_SIZE
CMIN=-15                       # CONSTANT_RANGE_MIN    
CMAX=15                        # CONSTANT_RANGE_MAX

ROWS=1
C=53                            # COLUMNS
la=40                           # ES_LAMBDA
F='vaf'                         # FITNESS_EVAL
EE='mae,mse,r2_score'           # EVALUATORS 

L1='INFO'						# loglevel
L2='S'							# number of files exported

echo $PROBLEM $FITNESS_EVAL "random seed: " $RS
python ../main.py -O $O -e $F -ee $EE -c $C -mg $mg -pm $mp -mc $mc -P $P -C $co $CMIN $CMAX -la $la -ps $ps -rs $RS1 -log $L1 $L2
# ten iterations with given param set
for i in {1..10..1}
do
echo "no seed " $i
python ../main.py -O $O -e $F -ee $EE -c $C -mg $mg -pm $mp -mc $mc -P $P -C $co $CMIN $CMAX -la $la -ps $ps -rs $RS2 -log $L1 $L2
done

echo "done"

