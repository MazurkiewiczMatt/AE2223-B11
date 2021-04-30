# A TU Delft 2nd Year Student Project (AE 2223-I)
## An Evaluation of Viability of FMCW Radar Sensors on MAVs
This repository is dedicated to an analysis of flight data collected via a MAV with the goal of concluding whether the radar used on board of the MAV is a viable solution for the Sense and Avoid system on MAVs. For most up-to-date information on the project and this repositery, please see the group's [Overleaf file](https://www.overleaf.com/read/mjbcnttmngxn).

## Scripts and Files
There are three main scripts, which are "range_vs_angle.py", "range_vs_time.py", and "tools.py".
* "range_vs_angle.py" creates a plot of the range between the radar and an obstacle against the angle measured to the obstacle. This script contains a slider as a feature
* "range_vs_time.py" creates a plot of the range between the radar and an obstacle against time. This script does not contain a slider as a feature
* "tools.py" contains the functions that are used within the two aforementioned scripts
* "Scrap.py" contains the previous version of the code, which is currently used as reference while working on the three main scripts

In terms of files, any files with the ".bag" extension are the input data files from the MAV's flights that are used by the scripts for analysis. To see an example of such data, please see CVS files inside of the folder titled "1".

## Troubleshooting 
At the moment of writing this text, the scripts are a work in progress and do not run without initial manipulations. 

## Contact
The project is ran and kept by the B11 student group. For questions, please do not hesitate to contact us via WhatsApp, MS Teams and Discord.
