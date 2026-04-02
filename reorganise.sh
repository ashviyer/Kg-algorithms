#!/bin/bash
# reorganise.sh
# Run from /root/Dreamwalk:  bash reorganise.sh

mkdir -p Dreamwalk_AOP

mv DREAMwalk README.md aop_raredisease_kgpathway_neo4j.csv atc_hierarchy.csv \
   data.csv dreamwalk_script.py dreamwalk_script_aop_09.py \
   preprocess_dreamwalk.py preprocess_dreamwalk_aop.py \
   requirements.txt setup_git.sh Dreamwalk_AOP/

git add .
git commit -m "Reorganise files into Dreamwalk_AOP folder"
git push
