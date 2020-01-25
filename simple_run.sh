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

P="keijzer_2"                 	# PROBLEM

ps=333 	                # POPULATION_SIZE
mg=111                       	# MAX_GENERATIONS
O='+,-,*,/,Sin,Cos,Pow2,Sqrt'	# USEDDOPERATORS
mp=0.16                        # MUTATION_PROBABILITY
mc=15                          # MUTATION_COUNTER
RS=23                          # RANDOM_SEED

co=0                           # CONSTANT_SIZE
CMIN=-15                       # CONSTANT_RANGE_MIN    
CMAX=15                        # CONSTANT_RANGE_MAX

ROWS=1
C=53                            # COLUMNS
la=40                           # ES_LAMBDA
F='vaf'                         # FITNESS_EVAL
EE='mae,mse,r2_score'           # EVALUATORS 

echo $PROBLEM $FITNESS_EVAL "random seed: " $RS
python ../main.py -O $O -e $F -ee $EE -c $C -mg $mg -pm $mp -mc $mc -P $P -co $co -cr $CMIN $CMAX -la $la -ps $ps -rs $RS
# ten iterations with given param set
for i in {1..10..1}
do
echo "no seed " $i
python ../main.py -O $O -e $F -ee $EE -c $C -mg $mg -pm $mp -mc $mc -P $P -co $co -cr $CMIN $CMAX -la $la -ps $ps
done

echo "done"

