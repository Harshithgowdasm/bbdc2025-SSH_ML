# bbdc2025-SSH_ML

Repository for [Bremen Big Data Challenge (BBDC)](https://bbdc.csl.uni-bremen.de/en/)

Description : see [required packages](https://github.com/Harshithgowdasm/bbdc2025-SSH_ML/blob/main/bbdc_2025_description_en.md)

### Team Members
- [Harshith Gowda](https://github.com/harshithgowdasm)
- [Sucheth Shenoy](https://github.com/sucheth17)
- [Selva](https://github.com/snachi2s)

---

### Setup Instructions

Clone the repo:
```sh
git clone https://github.com/Harshithgowdasm/bbdc2025-SSH_ML.git 
```

Download the task data and extract the folder in the base directory of the repo:
```sh
task
├── test_set
├── train_set
├── validation_set
├── Data_License_Agreement.txt
└── student_skeleton.csv
```

Create a python virtual environment and install the [required packages](https://github.com/Harshithgowdasm/bbdc2025-SSH_ML/blob/main/requirements.txt):
```sh
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### AutoGluon Model
Build machine learning solutions on raw data in a few lines of code. 

Documnetation : https://auto.gluon.ai/stable/index.html 

Used Model : https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html 