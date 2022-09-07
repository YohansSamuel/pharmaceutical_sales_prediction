# Pharmaceuticals Sales Prediction

<!-- Table of contents -->
- [Rossman Pharmaceuticals Sales Prediction](#Rossman-Pharmaceuticals-Sales-Prediction)
  - [About](#about)
  - [Objectives](#objectives)
  - [Data](#data)
  - [Repository overview](#repository-overview)
  - [Contrbutors](#contrbutors)
  - [License](#license)

## About
The finance team at Rossman Pharmaceuticals wants to forecast sales in all their stores across several cities six weeks ahead of time. Managers in individual stores rely on their years of experience as well as their personal judgment to forecast sales. 

The data team identified factors such as promotions, competition, school and state holidays, seasonality, and locality as necessary for predicting the sales across the various stores.

The task is to build and serve an end-to-end product that delivers this prediction to analysts in the finance team. 

## Objectives
To build a predictive model that can predict sales in all stores across all cities 6 weeks ahead of time.

## Data
The data used for this project could be foun in [Rossman Pharmaceuticals Sales Data](https://www.kaggle.com/c/rossmann-store-sales/data) dataset.

## Repository overview
 Structure of the repository:
 
        ├── models  (contains trained model)
        ├── .github  (github workflows for CI/CD, CML)
        ├── screenshots  (model versioning screenshots)
        ├── data    (contains data versioning metedata)
        ├── scripts (contains the main script)	
        │   ├── logger.py (logger for the project)
        │   ├── plot.py (handles plots)
        │   ├── preprocessing.py (dataset preprocessing)
        ├── notebooks	
        │   ├── EDA.ipynb (overview of the sales dataset)
        │   ├── preprocessing.ipynb (data preparation)
        ├── tests 
        │   ├── test_preprocessing.py (test for the preprocessing script)
        ├── README.md (contains the project description)
        ├── requirements.txt (contains the required packages)
        |── LICENSE (license of the project)
        └── .dvc (contains the dvc configuration)


## Contrbutor(s)
- Yohans Samuel

## License
[MIT](https://choosealicense.com/licenses/mit/)
