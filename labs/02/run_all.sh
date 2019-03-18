set -e

python mnist_training.py --optimizer=SGD --learning_rate=0.01
python mnist_training.py --optimizer=SGD --learning_rate=0.01 --momentum=0.9
python mnist_training.py --optimizer=SGD --learning_rate=0.1

python mnist_training.py --optimizer=Adam --learning_rate=0.001
python mnist_training.py --optimizer=Adam --learning_rate=0.01
python mnist_training.py --optimizer=Adam --decay=exponential --learning_rate=0.01 --learning_rate_final=0.001
python mnist_training.py --optimizer=Adam --decay=polynomial --learning_rate=0.01 --learning_rate_final=0.0001
