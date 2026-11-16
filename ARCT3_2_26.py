# ARCT (3,2,26) - Pure Python MLP from scratch (without any external libraries)
seed = 42
def random():
    global seed
    seed = (seed * 1664525 + 1013904223) % 4294967296
    return -0.4 + (seed / 4294967296) * 0.8

input1 = [0.0, 0.0, 1.0, 1.0, 0.5, 0.2]
input2 = [0.0, 1.0, 0.0, 1.0, 0.5, 0.8]
input3 = [1.0, 0.0, 0.0, 1.0, 0.5, 0.2]

output1 = [0.33, 0.33, 0.33, 1.0, 0.5, 0.4]
output2 = [0.0,  0.0,  0.0,  1.0, 1.0, 0.0]

lerningrate = 0.005

weight1 = random()
weight2 = random()
weight3 = random()
weight4 = random()
weight5 = random()
weight6 = random()
weight7 = random()
weight8 = random()
weight9 = random()
weight10 = random()
weight11 = random()
weight12 = random()
weight13 = random()
weight14 = random()
weight15 = random()
weight16 = random()
weight17 = random()
weight18 = random()
weight19 = random()
weight20 = random()

bias1 = 0
bias2 = 0
bias3 = 0
bias4 = 0
bias5 = 0
bias6 = 0

def relu(x): #If needed you can write the func in h1,h2,h3,h4 so you save a little computing time.
    if x < 0:        #Due to the size of the Network it would be pretty pointless and just make it more complicated
        return 0
    else:
        return x

for epoche in range(40000): #For my testing 40k was enough. But you can use more if needed.
    for inp1, inp2, inp3, out1, out2 in zip(input1, input2, input3, output1, output2):
        h1 = relu((inp1 * weight1) + (inp2 * weight2) + (inp3 * weight3) + bias1) #input neuron 1 hiddenlayer
        h2 = relu((inp1 * weight4) + (inp2 * weight5) + (inp3 * weight6) + bias2) #input neuron 2 hiddenlayer
        h3 = relu((inp1 * weight7) + (inp2 * weight8) + (inp3 * weight9) + bias3)
        h4 = relu((inp1 * weight10) + (inp2 * weight11) + (inp3 * weight12) + bias4)
        outpress1 = (h1 * weight13) + (h2 * weight14) + (h3 * weight15) + (h4 * weight16)+ bias5
        outpress2 = (h1 * weight17) + (h2 * weight18) + (h3 * weight19) + (h4 * weight20)+ bias6

        lossout1 = (outpress1 - out1)
        lossout2 = (outpress2 - out2)
        lossh1 = lossout1 * weight13 + lossout2 * weight17
        lossh2 = lossout1 * weight14 + lossout2 * weight18
        lossh3 = lossout1 * weight15 + lossout2 * weight19
        lossh4 = lossout1 * weight16 + lossout2 * weight20

        if h1 > 0:
            weight1 = weight1 - (lossh1 * inp1 * lerningrate)
            weight2 = weight2 - (lossh1 * inp2 * lerningrate)
            weight3 = weight3 - (lossh1 * inp3 * lerningrate)
            bias1 = bias1 - (lossh1 * lerningrate)
        if h2 > 0:
            weight4 = weight4 - (lossh2 * inp1 * lerningrate)
            weight5 = weight5 - (lossh2 * inp2 * lerningrate)
            weight6 = weight6 - (lossh2 * inp3 * lerningrate)
            bias2 = bias2 - (lossh2 * lerningrate)
        if h3 > 0:
            weight7 = weight7 - (lossh3 * inp1 * lerningrate)
            weight8 = weight8 - (lossh3 * inp2 * lerningrate)
            weight9 = weight9 - (lossh3 * inp3 * lerningrate)
            bias3 = bias3 - (lossh3 * lerningrate)
        if h4 > 0:
            weight10 = weight10 - (lossh4 * inp1 * lerningrate)
            weight11 = weight11 - (lossh4 * inp2 * lerningrate)
            weight12 = weight12 - (lossh4 * inp3 * lerningrate)
            bias4 = bias4 - (lossh4 * lerningrate)
        weight13 = weight13 - (lossout1 * h1 *  lerningrate)
        weight17 = weight17 - (lossout2 * h1 * lerningrate)
        weight14 = weight14 - (lossout1 * h2 * lerningrate)
        weight18 = weight18 - (lossout2 * h2 * lerningrate)
        weight15 = weight15 - (lossout1 * h3 *  lerningrate)
        weight19 = weight19 - (lossout2 * h3 * lerningrate)
        weight16 = weight16 - (lossout1 * h4 * lerningrate)
        weight20 = weight20 - (lossout2 * h4 * lerningrate)
        bias5 = bias5 - (lossout1 * lerningrate)
        bias6 = bias6 - (lossout2 * lerningrate)
print("--Training complete--")
while True:
    inp1 = float(input("Enter a number for input1: "))
    if inp1 == 0:
        break
    inp2 = float(input("Enter a number input2: "))
    inp3 = float(input("Enter a number input3: "))
    h1 = relu((inp1 * weight1) + (inp2 * weight2) + (inp3 * weight3) + bias1)
    h2 = relu((inp1 * weight4) + (inp2 * weight5) + (inp3 * weight6) + bias2)
    h3 = relu((inp1 * weight7) + (inp2 * weight8) + (inp3 * weight9) + bias3)
    h4 = relu((inp1 * weight10) + (inp2 * weight11) + (inp3 * weight12) + bias4)
    outpress1 = (h1 * weight13) + (h2 * weight14) + (h3 * weight15) + (h4 * weight16) + bias5
    outpress2 = (h1 * weight17) + (h2 * weight18) + (h3 * weight19) + (h4 * weight20) + bias6
    print(outpress1, outpress2)