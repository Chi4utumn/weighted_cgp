UNAME=$(uname)

# set path to python executeable
if [ "$UNAME" == "Linux" ] ; 
	then
	echo "linux"
. ../../../../experimental/.cgpvenv/bin/activate
	elif [[ "$UNAME" == CYGWIN* || "$UNAME" == MINGW* ]] ;
	then
	echo "windows"
. ../../../../experimental/.cgpvenv/Scripts/activate
	else
	echo "not supported os"
fi

# Get folder of bash script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

shopt -s nullglob
array=($DIR/*.json)

for json in "${array[@]}"
do
    echo $json
    python ../../main.py -j $json -log "INFO" "XXL"


done

echo "all done"
