rm -rf ./cartpole/

$PWD/../.venv/bin/bopt init \
        --param "batch_size:int:2:100" \
        --param "epochs:int:1:100" \
        --param "layers:int:2:10" \
        --param "units:int:2:50" \
        -C cartpole \
        --runner sge \
        $PWD/bopt_cartpole.sh
