ESE650 Project 2
Ekaterina Tolstaya

To run the filter code:
In filter.py, change the file_num variable to load a different data set.
Run filter.py. This will produce quaternion data in the filtered folder.

To generate panorama:
 - To run with vicon data - set mydata = 0
 - To run with my filter - set mydata = 1 and ensure the corresponding filtered data has been generated.

Change the file_num variable in pano.py to load the corresponding images.
Run pano.py and you'll see the final panorama.

Additional files:
- Quaternion - my class defining quaternion calculations
- util.py - some additional conversion functions for rotations and euler angles
- rotplot.py - included with the starter files
- filtered/ - directory containing intermediate filtered data
