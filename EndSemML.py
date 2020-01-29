#Import required modules
import numpy
import theano
import theano.tensor as T
rng = numpy.random

#load trainData and targets, get Tuple D, define number of training steps
train_data=numpy.load("titanic_train.npy")
train_supervision=numpy.load("train_targets.npy")
D=(train_data,train_supervision)
ncols=train_data.shape[1]
feats=ncols
training_steps = 40000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
scale = theano.function([], w, updates=[(w, w*0.001)]) #theano.function(inputs=[x,y],list of input variables 
                                                       #                outputs=..., what values to be returned
                                                       #                updates=...,  “state” values to be modified
                                                       #                givens=...,   substitutions to the graph)
print ("Initial model:")
scale()
print (w.get_value(), b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))       # Probability that target = 1
prediction = p_1 > 0.5                        # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()    # The cost to minimize
gw, gb = T.grad(cost, [w, b])                 # Compute the gradient of the cost
                                              

# Compile
train = theano.function(inputs=[x,y],outputs=[prediction, xent],updates=((w, w - 0.001 * gw), (b, b - 0.001 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print ("Final model:")
print (w.get_value(), b.get_value())

#Load Test Data
testData=numpy.load("titanic_test.npy")

#Predict Survival
testPassengerStartIndex=892
outputs = [[x[0]+testPassengerStartIndex,x[1]] for x in enumerate(predict(testData))]

#Prepare file for submission
numpy.savetxt("submission_th_1.csv", outputs, delimiter=',', fmt='%d,%d', header='PassengerId,Survived', comments = '')