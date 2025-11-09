10 rem Simple neural network forward pass in tensorBASIC
20 rem This demonstrates tensor operations for ML
30 rem
40 print "Neural Network Forward Pass"
50 print "==========================="
60 rem
70 rem Input layer (batch of 2 samples, 4 features)
80 dim x(2,4)
90 let x = [[1,2,3,4],[5,6,7,8]]
100 print "Input X (2x4):"
110 print x
120 rem
130 rem Weight matrix (4 features -> 3 hidden units)
140 dim w1(4,3)
150 let w1 = [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9],[1.0,1.1,1.2]]
160 print "Weights W1 (4x3):"
170 print w1
180 rem
190 rem Forward pass: hidden = X @ W1
200 let hidden = x @ w1
210 print "Hidden layer (2x3):"
220 print hidden
230 rem
240 rem Second layer weights (3 hidden -> 2 output)
250 dim w2(3,2)
260 let w2 = [[0.1,0.2],[0.3,0.4],[0.5,0.6]]
270 rem
280 rem Output layer
290 let output = hidden @ w2
300 print "Output (2x2):"
310 print output
320 rem
330 print "Forward pass complete!"
340 end
