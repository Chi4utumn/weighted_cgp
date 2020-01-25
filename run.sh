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


declare -a fit=('pearson' 'chisquare' 'mae' 'mse' 'r2_score')
declare -a ops=('+,-,*,/,Sin,Cos,Pow2,Sqrt' 
                'Sin,Cos,Pow2,Sqrt' 
                '+,-,*,/,Sin,Cos'
                '+,-,*,/,Pow2,Sqrt')
declare -a probs=('keijzer200' 'salustowicz_1997' 'gplearn' 'streeter_5')

# generations
for a in "${fit[@]}"
do
    # used operators
    for b in "${ops[@]}"
    do
        # number of columns
        for c in {10..100..10}
        do
            # maximum generations
            for k in {200..1000..100}
            do
                # mutation propability
                for l in 0.1 0.2 0.3 0.4 0.5
                do
                    # problems for eval
                    for p in "${probs[@]}"
                    do
                        python ./main.py -O $b -e $a -c $c -mg $k -pm $l -P $p
                    done
                done
            done
        done
    done
done

echo "all done"
