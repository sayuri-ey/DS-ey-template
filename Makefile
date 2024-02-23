.PHONY: tests

# AWS_ACCOUNT = 
# AWS_REGION = 
# PROJECT_NAME =
# AWS_ACCOUNT = ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
# DOCKER_IMAGE = $(AWS_ACCOUNT)/${PROJECT_NAME}


conda-create:
	conda create --name $(firstword $(subst -, ,$1)) python=$(word 2,$(subst -, ,$1))

update-time:
	sudo ntpdate pool.ntp.org
install:
	pip3 install -q -r requirements.txt
	virtualenv venv
update-requirements: # exemplo de uso: make update-requirements NEW="polar,nbconvert"
	echo "$(firstword $(NEW))"  | tr ',' '\n' >> requirements.txt
	pip install -r requirements.txt
	pip install pip-chill -q
	pip-chill > requirements.txt
activate:
	source ./venv/bin/activate
deactivate:
	deactivate
# install-docker:
# 	sudo dpkg --configure -a
# 	sudo apt-get install -f
# 	sudo apt-get update
# 	sudo apt-get upgrade -y
# 	curl -fsSL https://get.docker.com -o get-docker.sh
# 	sudo sh get-docker.sh
# 	sudo systemctl start docker
# 	sudo systemctl enable docker
# 	sudo usermod -aG docker $(USER)
# docker-push:
# 	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin $(AWS_ACCOUNT)
# 	docker build -t $(DOCKER_IMAGE) .
# 	docker push $(DOCKER_IMAGE)
# docker-pull:
# 	aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin $(AWS_ACCOUNT)
# 	docker pull $(DOCKER_IMAGE)
# update-requirements: # exemplo de uso: make update-requirements --NEW="polar,nbconvert"
# 	echo "$(firstword $(NEW))"  | tr ',' '\n' >> requirements.txt
# 	pip install -r requirements.txt
# 	pip install pip-chill -q
# 	pip-chill > requirements.txt
scripts:
	python3 -m jupyter nbconvert --to python --no-prompt 'dashboards/teste.ipynb'
	python3 -m jupyter nbconvert --to python --no-prompt 'pipeline/preprocess.ipynb'
	python3 -m jupyter nbconvert --to python --no-prompt 'pipeline/train.ipynb'
	python3 -m jupyter nbconvert --to python --no-prompt 'pipeline/evaluate.ipynb'
	python3 -m jupyter nbconvert --to python --no-prompt 'pipeline/interpret.ipynb'
	python3 -m jupyter nbconvert --to python --no-prompt 'pipeline/inference.ipynb'
# tests:
# 	docker run $(DOCKER_IMAGE) python -W ignore -m pytest -v -s tests/
# testcov:
# 	docker run $(DOCKER_IMAGE) python -m pytest --cov ./ --cov-report=term-missing tests/
# make teste:
# 	docker run $(DOCKER_IMAGE) python testes.py $(USER) $(PWD) $(SALIENT_USER) $(SALIENT_PASSWORD)
# make teste-python:
# 	python testes.py $(USER) $(PWD) $(SALIENT_USER) $(SALIENT_PASSWORD)

#nohup python -m src.salient.extract > salient_extraction.log 2>&1 &

#sudo lsof -i :5000

#ps aux