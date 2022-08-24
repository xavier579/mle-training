FILES=('.gitignore' 'setup.py')
for file in "${FILES[@]}"
do
    if test -f "$file"; then
        echo "$file exists."
    else
        touch $file
    fi
done

DIRECTORY=("logs" "data" "artifacts" "deploy" "src" "tests" "docs" "notebooks")
for dir in "${DIRECTORY[@]}"
do
    if [[ -d "$dir" ]]
    then
        echo "$dir exists on your filesystem."
    else
        mkdir $dir
        if [ $dir = 'data' ]
        then
            mkdir -p -- $dir/raw $dir/processed
            touch $dir/raw/.gitkeep $dir/processed/.gitkeep
        elif [ $dir = 'deploy' ]
        then
            mkdir -p -- $dir/conda $dir/docker
            touch $dir/conda/linux_cpu_py39.yml $dir/docker/Dockerfile
        elif [ $dir = 'src' ]
        then
            mkdir $dir/my_package
            touch $dir/my_package/__init__.py
        else
            touch $dir/.gitkeep
        fi
    fi
done