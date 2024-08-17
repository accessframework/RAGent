#!/bin/bash

export PIP_DISABLE_PIP_VERSION_CHECK=1
env_name="demoenv"

check_virtualenv() {
    if ! command -v virtualenv &> /dev/null; then
        echo "virtualenv is not installed. Installing..."
        python3 -m pip install --user virtualenv
        echo "virtualenv installation complete."
    fi
}

create_venv() {

    check_virtualenv

    if [ ! -d "$env_name" ]; then
        echo "Creating the environment $env_name and installing dependencies ..."
        python3 -m venv "$env_name"
        source "$env_name/bin/activate"

        pip install spacy --quiet
        pip install torch --quiet
        pip install -U spacy-experimental --quiet
        pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.1/en_coreference_web_trf-3.4.0a2-py3-none-any.whl --quiet
        pip install peft --quiet
        pip install transformers==4.30.2 --quiet
        pip install bitsandbytes --quiet
        pip install einops --quiet
        pip install nltk --quiet
        pip install captum --quiet
        pip install pandas --quiet
        pip install gdown==5.1.0 --quiet
        pip install click==8.1.7 --quiet

        python -m spacy download en_core_web_sm
    fi
}

install_gen_dep() {
  source "$env_name/bin/activate"
  pip install -U transformers --quiet
  pip install -U wheel
  pip install -U flash-attn
}

preprocess() {
    source "$env_name/bin/activate"
    python preprocess.py --policy_doc=${1:-"privacy_hotcrp.md"}
}

run_agentv() {
    source "$env_name/bin/activate"
    mkdir -p generations
    if [ -n "$1" ]
    then
        python demo.py --ent_file=$1
    else
        python demo.py 
    fi
    

}

create_venv
preprocess $1
install_gen_dep
run_agentv $2
