# EX--3-Implementation-of-MLP-for-a-non-linearly-separable-data
# NAME: LATHIKA K
# NO. 212224230140

# Aim:
To implement a perceptron for classification using Python
<H3>Theory:</H3>
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows:

XOR truth table
![Img1](https://user-images.githubusercontent.com/112920679/195774720-35c2ed9d-d484-4485-b608-d809931a28f5.gif)

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below

![Img2](https://user-images.githubusercontent.com/112920679/195774898-b0c5886b-3d58-4377-b52f-73148a3fe54d.gif)

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.To separate the two outputs using linear equation(s), it is required to draw two separate lines as shown in figure below:
![Img 3](https://user-images.githubusercontent.com/112920679/195775012-74683270-561b-4a3a-ac62-cf5ddfcf49ca.gif)
For a problem resembling the outputs of XOR, it was impossible for the machine to set up an equation for good outputs. This is what led to the birth of the concept of hidden layers which are extensively used in Artificial Neural Networks. The solution to the XOR problem lies in multidimensional analysis. We plug in numerous inputs in various layers of interpretation and processing, to generate the optimum outputs.
The inner layers for deeper processing of the inputs are known as hidden layers. The hidden layers are not dependent on any other layers. This architecture is known as Multilayer Perceptron (MLP).
![Img 4](https://user-images.githubusercontent.com/112920679/195775183-1f64fe3d-a60e-4998-b4f5-abce9534689d.gif)
The number of layers in MLP is not fixed and thus can have any number of hidden layers for processing. In the case of MLP, the weights are defined for each hidden layer, which transfers the signal to the next proceeding layer.Using the MLP approach lets us dive into more than two dimensions, which in turn lets us separate the outputs of XOR using multidimensional equations.Each hidden unit invokes an activation function, to range down their output values to 0 or The MLP approach also lies in the class of feed-forward Artificial Neural Network, and thus can only communicate in one direction. MLP solves the XOR problem efficiently by visualizing the data points in multi-dimensions and thus constructing an n-variable equation to fit in the output values using back propagation algorithm

<h3>Algorithm :</H3>

Step 1 : Initialize the input patterns for XOR Gate<BR>
Step 2: Initialize the desired output of the XOR Gate<BR>
Step 3: Initialize the weights for the 2 layer MLP with 2 Hidden neuron  and 1 output neuron<BR>
Step 3: Repeat the  iteration  until the losses become constant and  minimum<BR>
    (i)  Compute the output using forward pass output<BR>
    (ii) Compute the error<BR>
	(iii) Compute the change in weight ‘dw’ by using backward progatation algorithm. <BR>
    (iv) Modify the weight as per delta rule.<BR>
    (v)  Append the losses in a list <BR>
Step 4 : Test for the XOR patterns.

<H3>Program:</H3>
<pre>
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])
input_neurons = 2
hidden_neurons = 2
output_neurons = 1
samples = X.shape[1]
learning_rate = 0.1
epochs = 10000
np.random.seed(2)
W1 = np.random.rand(hidden_neurons, input_neurons)
W2 = np.random.rand(output_neurons, hidden_neurons)
loss_history = []
def sigmoid(value):
    return 1 / (1 + np.exp(-value))
def forward_pass(W1, W2, X):
    hidden_input = np.dot(W1, X)
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(W2, hidden_output)
    final_output = sigmoid(final_input)
    return hidden_input, hidden_output, final_input, final_output
def backward_pass(W1, W2, hidden_output, final_output, Y):
    error_output = final_output - Y
    grad_W2 = np.dot(error_output, hidden_output.T) / samples
    error_hidden = np.dot(W2.T, error_output) * hidden_output * (1 - hidden_output)
    grad_W1 = np.dot(error_hidden, X.T) / samples
    return grad_W1, grad_W2
for epoch in range(epochs):
    h_in, h_out, f_in, predictions = forward_pass(W1, W2, X)
    loss = -(1 / samples) * np.sum(
        Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions)
    )
    loss_history.append(loss)
    dW1, dW2 = backward_pass(W1, W2, h_out, predictions, Y)
    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()
def predict(W1, W2, test_input):
    _, _, _, output = forward_pass(W1, W2, test_input)
    output = np.squeeze(output)
    predicted_value = 1 if output >= 0.5 else 0
    print([i[0] for i in test_input], predicted_value)
print("Input  Output")
test = np.array([[1], [0]])
predict(W1, W2, test)
test = np.array([[1], [1]])
predict(W1, W2, test)
test = np.array([[0], [1]])
predict(W1, W2, test)
test = np.array([[0], [0]])
predict(W1, W2, test)
</pre>

<H3>Output:</H3>

<img width="677" height="547" alt="image" src="https://github.com/user-attachments/assets/d2309172-eadf-4257-8b71-d5d8fda3b485" />


<H3> Result:</H3>
Thus, XOR classification problem can be solved using MLP in Python 
