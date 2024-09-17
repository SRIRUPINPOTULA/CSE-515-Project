# CSE 515 - Multimedia and Web Databases 

# Title: Action Classification

# Data

The videos are divided into target and non_target videos. So, during the execution of the programs the input video file names are based on the location of the videos. 

`Example Usage: sample_input_1 = '/Users/srirupin/Downloads/target/cartwheel/Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0.avi'`

# Repo Directory Structure:
# 1.Code Directory: 

## 1.1 Task -1: 
In directory task-1, `Task-1.ipynb` file has been added, corresponding to the code and output for Task-1. In this Python file itself, the outputs of the sample inputs have been printed.

## 1.4 Task-4:  
In directory task-4, `Task-4.ipynb` file was added, corresponding to the code used to generate the output for all the videos of target videos. The outputs are exported to JSON which is part of `Output\Task-4\Task-1_Output`.

## 1.5 Task-5:  
In directory task-5, `Task-5.ipynb` file has been added that measures the `m=10` similar videos for the video file given using the JSON of Task-4 and using the logic of Task-1. The nearest search videos for the sample inputs have been printed there along with the distance


# 2. Output:

## 2.1 Task-1 Output: 
### 2.1.1 Features of Target Videos:
In the output directory, the directory `Task-4` contains the output of the code made for `Task-1_Output`. Each of the json files represents the target actions. The mapping of them to the target videos are listed below:

Output of action type sword => `sword_output.json`

Output of action type cartwheel => `cartwheel_output.json`

Output of action type drink => `drink_output.json`

Output of action type ride_bike => `ridebikeoutput.json`


## 2.2 Task-2 Output: 
### 2.2.1 Features of Target Video:
In the output directory, the directory `Task-4` contains the output of the code made for `Task-2_Output`. This directory consists of a directory of all the actions that are listed as target videos. And in each action directory for each input video, the output is stored as JSON.
