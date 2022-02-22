# Design Layout

Program needs to do the following things:

- Read multiple different dataset files and transform them to pandas arrays
- Transform panda arrays into 1 pandas array
- Train ML model on panda array, using mortality code as Label

Important to note:

- Mortality files are *.dat* files
- Other NHANSE files are *.xpt* files

pandas.align can be used to SQL join two dataframes
