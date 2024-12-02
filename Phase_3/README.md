# CSE 515 Phase 2 - Group 5

# Directory Structure:
The Phase-2 directory consists of the following directories

```plaintext
Phase_3/
│
├── README.md                   # Project Overview (this file)
│
├── Code/                       # Source Code for Phase 3
│   └── Util/                   # Source Code to assist Phase 3
│
├── Database/                   # Contains the db and json files
│
└── Report/                     # Contains the report for Phase-2
```

## 1. Code:
This directory consists of all the tasks' codes, including preprocessing and actual tasks.

### Util:

This Directory Consists of the corrected phase-1 code. 


KMeans.py: The python file consists of implementation code for KMeans, used as part of Task-1,2,3.

LDA.py: The python file consists of the LDA implementation that is part of Task-2.

PCA.py: The python file consists implementation of PCA that is used as part of Task-1,2,3.

SVD.py: The python file consists implementation of PCA that is used as part of Task-1,2,3.

KMeanslatentmodel.py: The python file consists implementation that gathers all the features of target video and used for calculation of latent models which are used for Task-1b.

services.py: This file contains the implementation to extract the features and reduce to latent models.

### Task-0a:

Task_0a.py: Consists code to find the inherent dimensionality.

### Task-1:

Task_1a.py: Implementation of Spectral Clustering.

Task_1b.py: Consists of code to implement KMeans clustering

### Task-2:

Task_2.py: Consists that code that implements `KNN` and `SVM` classifiers.

### Task-3: 

Task_3a.py: Consists of code that implements LSH for the given input.

Task-3b_preprocess.py: Consists of code that implements the thumbnails.

Task_3b.py: Consists of code that implements LSH for all the videos

## 2. Database:
This directory consists of all the json files that are stored as part of all the tasks and the database.

<br>
<br>

# How to Run

1. Download, Extract and copy the dataset to hmdb51_org and hmdb51_org_stips

[Download Link](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

2. Install the required dependencies:
> `pip install -r requirements.txt`

3. Run the Source Code from `Code` directory in order:
> `cd Code`
>
> `python Task_0a.py`
>
> `python Task_1a.py`
>
> `python Task_1b.py`
> 
> `python Task_2.py`
>
> `python Task-3a.py`
>
> `python Task-3b_preprocess.py`
>
> `python Task-3b.py`

<br>

# Using the SQLite Database

1. Connect to the database from the CLI
> `sqlite3 database/Phase_3.db`

2. List the tables in the db
> `.tables`

3. View the table Schema
> `.schema data`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;or

>`PRAGMA table_info(data)`

4. Use SQL to query db
> `select * from data limit 5;`

5. Other Resources

- [SQLite Browser](http://sqlitebrowser.org/)
- [SQLite Viewer](https://inloop.github.io/sqlite-viewer/)
- [spatialite-gui](https://www.gaia-gis.it/fossil/spatialite_gui/index)


