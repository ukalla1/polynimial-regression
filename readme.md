This project is the implementation of polynimoal linear regression in python using the scikit learn library.
The following are the goals of the code:
	1. To read a CSV file with the data inputs.
	2. Store the data into 2 variables based on dependent and independent vars (Here, it is assumed that the data has independent vars in all but the last column, with the data itself having 3 columns).
	3. Split the data into train and test data sets (80 percent data in train and 20 in test).
	4. Create the polynomial features.
	5. Create the polynimial regressor.
	6. Fit the train set to the regressor, and predict the outputs on the test set.
	7. Plot the outputs.