For the first assignment, we will revisit some of the core skills you reviewed in the first 3-4 lectures.

1. create a dataset with 10,000 rows and 4 random variables: 2 of them normally distributed, 2 uniformly distributed

2. add another variable ("y") as a linear combination, with some coefficients of your choice, of: the 4 variables above; the square of one of the variables; and some random "noise" (randomly distributed, centered at 0, small variance)

3. estimate the linear regression coefficients using OLS for the whole data

4. use bootstrapping to create 10 other samples from the data your created in #1-#2 above

5. estimate the linear regression coefficients using OLS for the 10 bootstrap samples in 4

6. for each coefficient in #3 compute the standard deviation from the mean of the 10 corresponding coefficients estimated in #5

Clearly mark each step in the script (with comments). Make sure the script runs from top to bottom without errors; points will be subtracted if the script is not working or needs editing to work. 

You may submit either a Jupyter Notebook or a python script.

