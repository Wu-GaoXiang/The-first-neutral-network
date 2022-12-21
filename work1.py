# Make Your Own Neutral Network
# code for three layer neutral network

import numpy
import scipy.special
import matplotlib

#neural network class definition
class neuralNetwork :

    #initialise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningingrate) :
        #set the number node in each input,hidden,output layer
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        #set the learning rate
        self.lr=learningingrate

        #setting the orignial weight form random normal distribution
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        self.wih = numpy.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes,self.inodes) )
        self.who = numpy.random.normal(0.0, pow(self.onodes,-0.5),(self.onodes,self.hnodes) )

        #activation function is the sigmoid function
        self.actication_function = lambda x: scipy.special.expit(x)

        pass


    #train the network
    def train(self,inputs_lists, targets_lists) :
        # convert input to 2d array
        inputs = numpy.array(inputs_lists,ndmin=2).T
        targets = numpy.array(targets_lists, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.actication_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #calculate signals emerging from final output layer
        final_outputs = self.actication_function(final_inputs)

        #output layer error is the (target-actual)
        output_errors = targets - final_outputs
        #hidden error is the output_errors, which is slipt by weights, recombined at hidden node
        hidden_errors = numpy.dot(self.who.T,output_errors)

        #update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)) , numpy.transpose(hidden_outputs))
        #update the links between the input and hidden output
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)) , numpy.transpose(inputs))

        pass


    #query the network
    def query(self,inputs_list) :
        #convert inputs list to 2d array,并且转置。
        inputs = numpy.array(inputs_list,ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)


        # calculate the signals emerging from hidden layer
        hidden_outputs = self.actication_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.actication_function(final_inputs)

        return final_outputs

        pass
    pass

#number of input,hidden,and output node
input_node=784
hidden_node=100
output_node=10

#learning rate is 0.3
learning_rate=0.3

#create instance of neutral network
#输入点、输出点、隐藏点均为3的neuralNetwork 被叫做n的名字，使用n这个名字输入
n = neuralNetwork(input_node,hidden_node,output_node,learning_rate)

#load the minist training data CSV file into a list
training_data_file = open("D:/mnist_dataset/mnist_train.csv",'r')        #这里用了六万个数据重复
training_data_list = training_data_file.readlines()
training_data_file.close()

#train the neutral network

#go through all recorders in the training data set
for record in training_data_list:
    #split the record by the ','commmas
    all_values = record.split(',')
    #scale and shift the inputs
    inputs = (numpy.asfarray(all_values [1:])/255.0 * 0.99 + 0.01)
    #create the target output the value (all in 0.01, and the desired number is 0.99)
    targets = numpy.zeros(output_node) + 0.01
    #all the all_value[0] is the target for recording
    targets[int(all_values[0]) ] = 0.99
    n.train(inputs,targets)
    pass

#load the mnist test data CSV into a list
test_data_file = open("D:/mnist_dataset/mnist_test.csv",'r' )         #这里的test也是真的test
test_data_list = test_data_file.readlines()
training_data_file.close()

#get the first test record
all_values = test_data_list[0].split(',')
print(all_values[0])

#  print( n.query((numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01) )


# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []
# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # print(label, "network's answer")                                                Here is the network's answer
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to sorecard
        scorecard.append(0)
        pass

    pass

#calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print("performace = ", scorecard_array.sum() /scorecard_array.size)