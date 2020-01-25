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

declare -a EVALUATOR=('vaf' 'mae' 'mse' 'r2_score')
declare -a OPERATORS=('+,-,*,/'
                      '+,-,*,/,Sin,Cos,Pow2,Sqrt,exp,log'
                    )

P="keijzer_2"             # PROBLEM
EE='mae,mse,r2_score,vaf' # EVALUATORS
CMIN=-15                  # constant range min
CMAX=15                   # constant range max
RS1=23                          # RANDOM_SEED
RS2=-1							# RANDOM RANDOM_SEED

L1='INFO'						# loglevel
L2='S'							# number of files exported


# generations
for F in "${EVALUATOR[@]}"
do
    # used operators
    for O in "${OPERATORS[@]}"
    do
        # number of columns
        for C in 53
        do
            # maximum generations
            for mg in 111
            do
                # mutation propability
                for mp in 0.16
                do
                    # variation of number of constants
                    for co in 0 1 2
                    do

                    # variation of population size 
                    for ps in 333
                    do
                    # variation of lambdas 
                    for la in 40
                    do
                    # mutation counters 
                    for mc in 15
                    do

# eval with fixed random seed for every fitness evaluator
echo $ops $fit $EE $c $mg $pm $mc $P $co $CMIN $CMAX $la $RS
python ../main.py -O $O -e $F -ee $EE -c $C -mg $mg -pm $mp -mc $mc -P $P -C $co $CMIN $CMAX -la $la -ps $ps -rs $RS1 -log $L1 $L2
                        # ten iterations with given param set
                        for i in {1..10..1}
                        do
                        echo $i
python ../main.py -O $O -e $F -ee $EE -c $C -mg $mg -pm $mp -mc $mc -P $P -C $co $CMIN $CMAX -la $la -ps $ps -rs $RS2 -log $L1 $L2
                        done
                        echo "done"

                    done
                    done
                    done
                    done
                done
            done
        done
    done
done
echo "all done"
