import matplotlib.pyplot as plt
from reg_utils import load_2D_dataset, predict, predict_dec, plot_decision_boundary
from testCases import compute_cost_with_regularization_test_case, backward_propagation_with_regularization_test_case
from testCases import forward_propagation_with_dropout_test_case, backward_propagation_with_dropout_test_case
from DNN import model, compute_cost_with_regularization, backward_propagation_with_regularization
from DNN import forward_propagation_with_dropout, backward_propagation_with_dropout

# set default size of plots
# plt.rcParams['figure.figsize'] = (7.0, 4.0)
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# If the dot is blue, it means the French player managed to hit the ball with his/her head
# If the dot is red, it means the other team's player hit the ball with their head
train_X, train_Y, test_X, test_Y = load_2D_dataset()
# plt.show()


# train the model without any regularization, and observe the accuracy on the train/test sets.
'''
parameters = model(train_X, train_Y)
print("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

#The non-regularized model is obviously overfitting the training set. It is fitting the noisy points! Lets now look at two techniques to reduce overfitting.
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
'''

# test case compute_cost_with_regularization
'''
A3, Y_assess, parameters = compute_cost_with_regularization_test_case()
print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd=0.1)))


#test case backward_propagation_with_regularization
X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()
grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd=0.7)
print("dW1 = " + str(grads["dW1"]))
print("dW2 = " + str(grads["dW2"]))
print("dW3 = " + str(grads["dW3"]))

#not overfitting the training data anymore because of L2 regularization method
parameters = model(train_X, train_Y, lambd=0.7)
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
'''

X_assess, parameters = forward_propagation_with_dropout_test_case()

A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob=0.7)
print("A3 = " + str(A3))

X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()

gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob=0.8)

print("dA1 = " + str(gradients["dA1"]))
print("dA2 = " + str(gradients["dA2"]))

parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3)

print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)