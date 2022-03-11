# Design Layout

Program needs to do the following things:

- Read multiple different dataset files and transform them to pandas arrays
- Transform panda arrays into 1 pandas array
- Train ML model on panda array, using mortality code as Label

Important to note:

- Mortality files are *.dat* files
- Other NHANSE files are *.xpt* files

pandas.align can be used to SQL join two dataframes

## Experiments + Variable Combinations

Experiment1 = Lab Work -> CVR
Experiment2 = Classic Heart attach symptoms -> CVR
Experiment3 = Lab + Classic -> CVR

All measurements in (mg/dl)

| Year  | Total_Chol | LDL_Chol | HDL_Chol | Glucose (FBG) | Triglyceride |
| ----  | ---------  | -------- | -------- | ------------- | ------------ |
| 99-00 |      LBXTC | LBDLDL   | LBDHDL   | LBXSGL        | LBXTR        |
| 01-02 |      LBXTC | LBDLDL   | LBDHDL   | LB2GLU        | LBXTR        |
| 03-04 |      LBXTC | LBDLDL   | LBXHDD   | LBXGLU        | LBXTR        |
| 05-06 |      LBXTC | LBDLDL   | LBDHDD   | LBXGLU        | LBXTR        |
| 07-08 |      LBXTC | LBDLDL   | LBDHDD   | LBXGLU        | LBXTR        |
| 09-10 |      LBXTC | LBDLDL   | LBDHDD   | LBXGLU        | LBXTR        |

Info on what variables are available in NHANSE for experiment2

| Year  | Diabetes  | Hyperten  | Chest Pain | Standing Relieve pain | Pain in left arm | Severe pain in chest |
| ----  | --------- | --------  | ---------- | --------------------- | ---------------- | -------------------- |
| 99-00 | DIABETES | HYPERTEN   | Missing    | CDQ005                | CDQ009G          | CDQ008               |
| 01-02 | DIABETES | HYPERTEN   | CDQ001     | CDQ005                | CDQ009G          | CDQ008               |
| 03-04 | DIABETES | HYPERTEN   | CDQ001     | CDQ005                | CDQ009G          | CDQ008               |
| 05-06 | DIABETES | HYPERTEN   | CDQ001     | CDQ005                | CDQ009G          | CDQ008               |
| 07-08 | DIABETES | HYPERTEN   | CDQ001     | CDQ005                | CDQ009G          | CDQ008               |
| 09-10 | DIABETES | HYPERTEN   | CDQ001     | CDQ005                | CDQ009G          | CDQ008               |


DIABETES      HYPERTEN    CHEST_PAIN  STANDING_RELIEVE  PAIN_LEFT_ARM  SEVERE_CHEST_PAIN
count  52185.000000  52185.000000  52185.000000      52185.000000   52185.000000       52185.000000
mean       0.012705      0.016461      0.594481          0.031005       0.012072           0.170892
std        0.111998      0.127240      0.870158          0.241672       0.290453           0.547351
min        0.000000      0.000000      0.000000          0.000000       0.000000           0.000000
25%        0.000000      0.000000      0.000000          0.000000       0.000000           0.000000
50%        0.000000      0.000000      0.000000          0.000000       0.000000           0.000000
75%        0.000000      0.000000      1.000000          0.000000       0.000000           0.000000
max        1.000000      1.000000      9.000000          9.000000       7.000000           9.000000
