# python_test

This module tries to predict moments in the Forex market that precede larger drops or increments of the prices.
The model is based on a Convolutional Neural Network network which tries to classify the input into 3 categories - BUY, SELL, and WAIT. The input for the model is an 3D vector with dimensions corresponding to:
1. **time** - I experimented with periods various number of days
2. **exchange pair** - various exchange pairs were compared, correlation among them was measured, 3 were picked for the input
3. **OLHC values** - Open, Low, High, and Close values for all the times and all the pairs

Since the classes are imbalanced, they are weighted according to the frequency of their occurrence. I experimented with thresholds for the percentual changes in the price for the class category estimation - when small percentual changes were taken for classifying into the BUY/SELL classes, there were more of them an more trandes could be executed, however, when commisions were taken into account, such trades would not be proffitable.
