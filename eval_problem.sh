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

declare -a EVALUATOR=('vaf' 'mae' 'mse' 'r2_score')
declare -a OPERATORS=('+,-,*,/'
                '+,-,*,/,Sin,Cos,Pow2,Sqrt' 
                'Sin,Cos,Pow2,Sqrt' 
                '+,-,*,/,Sin,Cos'
                '+,-,*,/,Pow2,Sqrt'
                'Pow2,Sqrt,exp,log'
                '+,-,*,/,Sin,Cos,Pow2,Sqrt,exp,log'
                'exp,log,Pow2,Sqrt'
                )

P="keijzer_4"             # PROBLEM
EE='mae,mse,r2_score,vaf' # EVALUATORS
CMIN=-15                  # constant range min
CMAX=15                   # constant range max
RS=23                     # random seed
PS=333

# generations
for fit in "${EVALUATOR[@]}"
do
    # used operators
    for ops in "${OPERATORS[@]}"
    do
        # number of columns
        for c in 65
        do
            # maximum generations
            for mg in 111
            do
                # mutation propability
                for mp in 0.16
                do
                    # variation of number of constants
                    for co in 2
                    do

                    # variation of lambdas 
                    for la in 40
                    do

                    # mutation counters 
                    for mc in 15
                    do

# eval with fixed random seed for every fitness evaluator
echo $ops $fit $EE $c $mg $pm $mc $P $co $CMIN $CMAX $la $RS
python ./main.py -O $ops -e $fit -ee $EE -c $c -mg $mg -pm $mp -mc $mc -P $P -co $co -cr $CMIN $CMAX -la $la -rs $RS -ps $PS

                        # ten iterations with given param set
                        for i in {1..10..1}
                        do
                        echo $i
python ./main.py -O $ops -e $fit -ee $EE -c $c -mg $mg -pm $mp -mc $mc -P $P -co $co -cr $CMIN $CMAX -la $la  -ps $PS
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
echo "all done"
