#!/usr/bin/env python3

# DayInYear: 1st of January=1, 2nd of January=2, ...
# DayOfWeek: Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
# UScoast: Non-coast=0, WestCoast=1, EastCoast=2

import numpy as np

verbose = True
N_verbose = 30
filename = 'data_UfoSightings.txt'

date = []
HourInDay = []
DayInYear = []
DayOfWeek = []
USstate = []
UScoast = []
Shape = []
DurationInSec = []


with open(filename, 'r' ) as infile:
    header1 = infile.readline()     # Read (and ignore) the first four line, which are header lines!
    header2 = infile.readline()
    header3 = infile.readline()
    header4 = infile.readline()
    if verbose:
        print('Date            HourInDay       DayInYear       DayOfWeek       USstate         UScoast         Shape           DurationInSec')
        
    counter = 0
    for line in infile:
        
        columns = line.strip().split()
        if (len(columns) == 8):
            
            date.append(columns[0])
            HourInDay.append(float(columns[1]))
            DayInYear.append(int(columns[2]))
            DayOfWeek.append(int(columns[3]))
            USstate.append(columns[4])
            UScoast.append(int(float(columns[5])))
            Shape.append(columns[6])
            DurationInSec.append(float(columns[7]))
            
            counter += 1
            
            if verbose and counter < N_verbose:
                print(f"""{date[-1]} \t{HourInDay[-1]:.1f} \t\t{DayInYear[-1]} \t\t{DayOfWeek[-1]} \t\t{USstate[-1]} \t\t{UScoast[-1]} \t\t{Shape[-1]:10s} \t{DurationInSec[-1]}""")

print(f"The total number of data points read is: {counter:d}")


date = np.array(date)
HourInDay = np.array(HourInDay)
DayInYear = np.array(DayInYear)
DayOfWeek = np.array(DayOfWeek)
USstate = np.array(USstate)
UScoast = np.array(UScoast)
Shape = np.array(Shape)
DurationInSec = np.array(DurationInSec)

