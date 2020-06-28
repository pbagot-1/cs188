import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        #print(nn.as_scalar(self.run(x)))
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        
        return -1
        

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        trained = True
        for x, y in dataset.iterate_once(1):
            if self.get_prediction(x) != nn.as_scalar(y):
                trained = False
                if (self.get_prediction(x) == -1):
                    self.w.update(x, 1)
                else:
                    self.w.update(x, -1)
                
        if not trained:
            self.train(dataset)
            
        
        "*** YOUR CODE HERE ***"

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.multiplier = -.01
        self.batch_size = 10
        self.m = nn.Parameter(1, 20)
        self.b = nn.Parameter(1, 20)
        
        self.m1 = nn.Parameter(20, 20)
        self.b1 = nn.Parameter(1, 20)
        
        self.m2 = nn.Parameter(20, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        print(x)
        xm = nn.Linear(x, self.m)
        print(xm)
        addBias = nn.AddBias(xm, self.b)
        print(addBias)
        addBias = nn.ReLU(addBias)
        print(addBias)
        mulFirst = nn.Linear(addBias, self.m1)
        addFirstBias = nn.AddBias(mulFirst, self.b1)
        addFirstBias = nn.ReLU(addFirstBias)
        mulSecond = nn.Linear(addFirstBias, self.m2)
        print(mulSecond)
        addSecondBias = nn.AddBias(mulSecond, self.b2)
        
        return addSecondBias

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred = self.run(x)
        loss = nn.SquareLoss(pred, y)
        return loss


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        
        for x, y in dataset.iterate_forever(self.batch_size):
            print(x)
            print("what")
            lossNode = self.get_loss(x, y)
            if nn.as_scalar(lossNode) < .001:
                return
            grad_wrt_m, grad_wrt_b, grad_wrt_m1, grad_wrt_b1,grad_wrt_m2, grad_wrt_b2 = nn.gradients(lossNode, [self.m, self.b, self.m1, self.b1, self.m2, self.b2])
            self.m.update(grad_wrt_m, self.multiplier)
            self.b.update(grad_wrt_b, self.multiplier)
            self.m1.update(grad_wrt_m1, self.multiplier)
            self.b1.update(grad_wrt_b1, self.multiplier)
            self.m2.update(grad_wrt_m2, self.multiplier)
            self.b2.update(grad_wrt_b2, self.multiplier)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.multiplier = -.15
        self.batch_size = 20
        self.m = nn.Parameter(784, 200)
        self.b = nn.Parameter(1, 200)
        
        self.m1 = nn.Parameter(200, 200)
        self.b1 = nn.Parameter(1, 200)
        
        
        
        self.m2 = nn.Parameter(200, 200)
        self.b2 = nn.Parameter(1, 200)
        
              
        self.m3 = nn.Parameter(200, 10)
        self.b3 = nn.Parameter(1, 10)
        
              
        self.m4 = nn.Parameter(20, 20)
        self.b4 = nn.Parameter(1, 20)
        
              
        self.m5 = nn.Parameter(20, 10)
        self.b5 = nn.Parameter(1, 10)
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #print(x)
        xm = nn.Linear(x, self.m)
        #print(xm)
        addBias = nn.AddBias(xm, self.b)
        #print(addBias)
        addBias = nn.ReLU(addBias)
        #print(addBias)
        mulFirst = nn.Linear(addBias, self.m1)
        addFirstBias = nn.AddBias(mulFirst, self.b1)
        addFirstBias = nn.ReLU(addFirstBias)
        mulSecond = nn.Linear(addFirstBias, self.m2)
        #print(mulSecond)
        addSecondBias = nn.AddBias(mulSecond, self.b2)
        addSecondBias = nn.ReLU(addSecondBias)
        
        a = nn.Linear(addSecondBias, self.m3)
        b = nn.AddBias(a, self.b3)
        #c = nn.ReLU(b)
        
        
        """d = nn.Linear(c, self.m4)
        e = nn.AddBias(d, self.b4)
        f = nn.ReLU(e)
        
        g = nn.Linear(f, self.m5)
        print(mulSecond)
        h = nn.AddBias(g, self.b5)"""
        return b

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        output = self.run(x)
        return nn.SoftmaxLoss(output , y)
        
        """nn.SoftmaxLoss computes a batched softmax loss, used for classification problems.

    Usage: nn.SoftmaxLoss(logits, labels), where logits and labels both have shape batch_size×num_classes

. The term “logits” refers to scores produced by a model, where each entry can be an arbitrary real number. The labels, however, must be non-negative and have each row sum to 1. Be sure not to swap the order of the arguments!"""

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_once(self.batch_size):
            #print(x)
            #print("what")
            lossNode = self.get_loss(x, y)
            grad_wrt_m, grad_wrt_b, grad_wrt_m1, grad_wrt_b1,grad_wrt_m2, grad_wrt_b2, grad_wrt_m3, grad_wrt_b3 = nn.gradients(lossNode, [self.m, self.b, self.m1, self.b1, self.m2, self.b2, self.m3, self.b3])
            self.m.update(grad_wrt_m, self.multiplier)
            self.b.update(grad_wrt_b, self.multiplier)
            self.m1.update(grad_wrt_m1, self.multiplier)
            self.b1.update(grad_wrt_b1, self.multiplier)
            self.m2.update(grad_wrt_m2, self.multiplier)
            self.b2.update(grad_wrt_b2, self.multiplier)
            self.m3.update(grad_wrt_m3, self.multiplier)
            self.b3.update(grad_wrt_b3, self.multiplier)
            """self.m4.update(grad_wrt_m4, self.multiplier)
            self.b4.update(grad_wrt_b4, self.multiplier)
            self.m5.update(grad_wrt_m5, self.multiplier)
            self.b5.update(grad_wrt_b5, self.multiplier)"""
           
        if dataset.get_validation_accuracy() < .975:
            self.train(dataset)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.multiplier = -.01
        self.batch_size = 20
        self.firstMatrix = nn.Parameter(self.num_chars, 350)
        self.b = nn.Parameter(1, 350)
        self.b2 = nn.Parameter(1, 5)
        self.secondMatrix = nn.Parameter(350, 350)
        self.getScores = nn.Parameter(350, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #print(len(xs))
        z = nn.Linear(xs[0], self.firstMatrix)
        z = nn.ReLU(z)
        for xi in xs[1:len(xs)]:
            addBias = nn.AddBias(nn.Linear(z, self.secondMatrix), self.b)
            z = nn.Add(nn.Linear(xi, self.firstMatrix), addBias)
            z = nn.ReLU(z)
            
        return nn.AddBias(nn.Linear(z, self.getScores), self.b2)
        

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        run = self.run(xs)
        loss = nn.SoftmaxLoss(run, y)
        return loss
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_once(self.batch_size):
            lossNode = self.get_loss(x, y)
            grad_wrt_w1, grad_wrt_w2, grad_wrt_w3, grad_wrt_b, grad_wrt_b2 = nn.gradients(lossNode, [self.firstMatrix, self.secondMatrix, self.getScores, self.b, self.b2])
            self.firstMatrix.update(grad_wrt_w1, self.multiplier)
            self.secondMatrix.update(grad_wrt_w2, self.multiplier)
            self.getScores.update(grad_wrt_w3, self.multiplier)
            self.b.update(grad_wrt_b, self.multiplier)
            self.b2.update(grad_wrt_b2, self.multiplier)
        if dataset.get_validation_accuracy() < .86:
            self.train(dataset)    
            
        
