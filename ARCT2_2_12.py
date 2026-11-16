# ARCT (2,2,12) - Pure Python MLP from scratch (without any external libraries)
seed = 42
def random():
    global seed
    seed = (seed * 1664525 + 1013904223) % 4294967296
    return -0.4 + (seed / 4294967296) * 0.8

Input1 = [1.0, 1.0, 3.0, 3.0]
Input2 = [0, 0.4, 0, 0.4]
cool = [1.0, 0.6, 0, 0]
heat = [0, 0 , 0.6, 1.0]

lerningrate = 0.01

weight1 = random() #The weights always start with the same values for repeatable tests.
weight2 = random() #It's advised to never use the same number for multiple weights, because then every neuron does the same
weight3 = random() #Start with different numbers for each weight
weight4 = random()
weight5 = random()
weight6 = random()
weight7 = random()
weight8 = random()

bias1 = 0 #Biases normally start at a value = 0
bias2 = 0
bias3 = 0
bias4 = 0

def relu(x):
    if x < 0:
        return 0
    else:
        return x

for time in range(40000): #This value depends on your input parameters but 40k should be more then enough
    for inp1, inp2, k, h in zip(Input1, Input2, cool, heat):
        h1 = relu((inp1 * weight1) + (inp2 * weight2) + bias1) #input neuron 1 hiddenlayer
        h2 = relu((inp1 * weight3) + (inp2 * weight4) + bias2) #input neuron 2 hiddenlayer
        out1 = h1 * weight5 + h2 * weight6 + bias3
        out2 = h1 * weight7 + h2 * weight8 + bias4
        lossout1 = (out1 - k)
        lossout2 = (out2 - h)
        lossh1 = lossout1 * weight5 + lossout2 * weight7
        lossh2 = lossout1 * weight6 + lossout2 * weight8
        if h1 > 0:                                          #Due to using Relu funktion we only train the weights and biases if the own neuron value is greater then zero.
            weight1 = weight1 - (lossh1 * inp1 * lerningrate)
            weight2 = weight2 - (lossh1 * inp2 * lerningrate)
            bias1 = bias1 - (lossh1 * lerningrate)
        if h2 > 0:
            weight3 = weight3 - (lossh2 * inp1 * lerningrate)
            weight4 = weight4 - (lossh2 * inp2 * lerningrate)
            bias2 = bias2 - (lossh2 * lerningrate)
        weight5 = weight5 - (lossout1 * h1 *  lerningrate)
        weight6 = weight6 - (lossout1 * h2 * lerningrate)
        weight7 = weight7 - (lossout2 * h1 * lerningrate)
        weight8 = weight8 - (lossout2 * h2 * lerningrate)
        bias3 = bias3 - (lossout1 * lerningrate)
        bias4 = bias4 - (lossout2 * lerningrate)
print(weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, bias1, bias2, bias3, bias4)
inp1 = float(input("Enter a number for input1: "))
inp2 = float(input("Enter a number for input2: "))
h1 = relu((inp1 * weight1) + (inp2 * weight2) + bias1)  # input neuron 1 hiddenlayer
h2 = relu((inp1 * weight3) + (inp2 * weight4) + bias2)  # input neuron 2 hiddenlayer
out1 = h1 * weight5 + h2 * weight6 + bias3
out2 = h1 * weight7 + h2 * weight8 + bias4
print(f" Output 1: {out1} and output 2: {out2}")


