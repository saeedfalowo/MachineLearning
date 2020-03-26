# Implement the following functions

# Activation (sigmoid) function
def sigmoid(x):
    """
    Sigmoid activation function
    𝜎(𝑥)=1/(1+𝑒−𝑥)
    """
    sig = 1 / (1 + np.exp(-x))
    return sig

# Output (prediction) formula
def output_formula(features, weights, bias):
    """
    Output (prediction) formula
    y_hat =𝜎(𝑤1𝑥1+𝑤2𝑥2+𝑏)
    """
    pred = sigmoid(np.dot(features, weights) + bias)
    return pred

# Error (log-loss) formula
def error_formula(y, output):
    """
    Error function
    𝐸𝑟𝑟𝑜𝑟(𝑦,y_hat )=−𝑦log(y_hat)−(1−𝑦)log(1−y_hat)
    """
    error = - y*np.log(output) - (1 - y) * np.log(1-output)
    return error

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    """
    The function that updates the weights
    𝑤𝑖⟶𝑤𝑖+𝛼(𝑦−y_hat)𝑥𝑖
    
    𝑏⟶𝑏+𝛼(𝑦−y_hat)
    """
    # Use output function to calculate the output
    output = output_formula(x,weights,bias)
    delta_error = y - output
    weights += learnrate * delta_error * x # 𝑤𝑖⟶𝑤𝑖+𝛼(𝑦−y_hat)𝑥𝑖
    bias += learnrate * delta_error        # 𝑏⟶𝑏+𝛼(𝑦−y_hat)
    return weights, bias