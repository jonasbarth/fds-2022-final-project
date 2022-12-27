# Final Project
Final Project for Fundamentals of Data Science 2022. The project aim is to classify Landsat 7 satellite images from the
Hindu Kush Himalaya region into glacier and non-glacier.

# Contributors
* Nemish Murawat
* Matteo Migliarini
* Mattia Castaldo
* Javier Martinez
* Jonas Barth

# Report
We produced a [report](doc/FDS_Final_Project_Report.pdf) where we summarise our findings.

# Inspiration
[Paper](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2020/57/paper.pdf) that we base the project on.

# Data
AWS S3 Links for the raw data that we used:
* [dev](https://fds-final-project.s3.amazonaws.com/dev.zip)
* [test](https://fds-final-project.s3.amazonaws.com/test.zip)
* [train](https://fds-final-project.s3.amazonaws.com/train.zip)

# Structure
* `custom` - contains custom pipeline functionalities.
* `dataset` - contains metadata.
* `model` - contains the models we used.
* `preprocessing` - contains logic for preprocessing.
* `report` - contains scripts for generating visuals for the report.
* `script` - contains some scripts for data exploration.
* `util` - utility functions.

# Installing Requirements
```
pip install -r requirements.txt
```
