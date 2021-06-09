# A TU Delft 2nd Year Student Project (AE 2223-I)
## An Evaluation of Obstacle Detection Preformance of 24-GHz FMCW Radar Sensors on MAVs
This repository is dedicated to an analysis of flight data collected via a MAV with the goal of concluding whether the radar used on board of the MAV is a viable solution as part of the Sense and Avoid system on MAVs responsible for obstacle detection. For most up-to-date information on the project and this repository, please see the group's [Overleaf file](https://www.overleaf.com/read/mjbcnttmngxn).

## Scripts and Files
There are four main scripts, which are "range_vs_angle.py", "range_vs_time.py", "error_vs_bag.py" and "tools.py".
* "range_vs_angle.py" creates a plot of the range between the radar and an obstacle against the angle measured to the obstacle. This script contains a slider as a feature
* "range_vs_time.py" creates a plot of the range between the radar and an obstacle against time. This script does not contain a slider as a feature
* "error_vs_bags.py" creates a plot of the range, angle and velocity error for all bag files
* "tools.py" contains the functions that are used within the two aforementioned scripts
* "Scrap.py" contains the previous version of the code, which is currently used as reference while working on the three main scripts
* "pose_optitrack.py" creates a plot of a single flight trajectory with the 3 obstacle positions

In terms of files, any files with the ".bag" extension are the input data files from the MAV's flights that are used by the scripts for analysis. To see an example of such data, please see CVS files inside of the folder titled "1".

## Troubleshooting 
At the moment, all scripts run successfully. 
In order to replicate the figures from the team's paper, however, certain manipulations are needed from the coder. ROS bag 41 was used for making figures found in the Methodolgy section. "Sample 1", "Sample 2", and "Sample 3" from the Results section are ROS bags 10, 40 and 70, respectively. 
It is also important to note that to achieve certain images the originals, obtained via Matplotlib, were post-processed using visual editing softwares, like InkScape.  

## Contact
The project is run and kept by the B11 student group. For questions, please do not hesitate to contact us via WhatsApp, MS Teams and Discord. 
