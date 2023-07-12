install:
	pip3 install -q -r requirements.txt
	virtualenv venv
activate:
	source ./venv/bin/activate
deactivate:
	deactivate
scripts:
	python3 -m jupyter nbconvert --to python --no-prompt 'pipeline/preprocess.ipynb'
	python3 -m jupyter nbconvert --to python --no-prompt 'pipeline/train.ipynb'
	python3 -m jupyter nbconvert --to python --no-prompt 'pipeline/evaluate.ipynb'
	python3 -m jupyter nbconvert --to python --no-prompt 'pipeline/interpret.ipynb'
	python3 -m jupyter nbconvert --to python --no-prompt 'pipeline/inference.ipynb'