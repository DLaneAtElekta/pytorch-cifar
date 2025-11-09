10 rem Logistic Regression with Gradient Descent
20 rem Binary classification example using tensorBASIC gradients
30 rem This demonstrates:
40 rem - Creating trainable parameters with param()
50 rem - Forward pass computation
60 rem - Loss calculation
70 rem - Backward pass with .backward()
80 rem - Gradient access with .grad
90 rem - Manual parameter updates with detach()
100 rem - Gradient zeroing with .zero_grad()
110 rem
120 print "Logistic Regression Training Example"
130 print "===================================="
140 print ""
150 rem
160 rem Create simple 2D training data
170 rem Class 0: points near [0,0]
180 rem Class 1: points near [1,1]
190 dim x_data(4, 2)
200 let x_data = [[0,0],[0,1],[1,0],[1,1]]
210 dim y_data(4, 1)
220 let y_data = [[0],[0],[0],[1]]
230 rem
240 print "Training data: 4 samples, 2 features"
250 print "Task: AND function (output 1 only when both inputs are 1)"
260 print ""
270 rem
280 rem Initialize model parameters (2 inputs -> 1 output)
290 rem y = sigmoid(x @ w + b)
300 let w = param(2, 1)
310 let b = param(1, 1)
320 rem
330 print "Model initialized: Linear -> Sigmoid"
340 print "Parameters: w (2x1), b (1x1)"
350 print ""
360 rem
370 rem Hyperparameters
380 let lr = 0.1
390 let epochs = 50
400 rem
410 print "Training for 50 epochs with learning rate 0.1"
420 print ""
430 rem
440 rem Training loop
450 for epoch = 1 to epochs step 1
460   rem Forward pass
470   let logits = x_data @ w + b
480   rem
490   rem For simplicity, use MSE loss instead of BCE
500   rem (BCE would require sigmoid implementation)
510   let y_pred = logits
520   rem
530   rem Compute loss
540   let diff = y_pred - y_data
550   let loss = diff.mean()
560   rem
570   rem Print every 10 epochs
580   let check = epoch - 10 * (epoch / 10)
590   if check == 0 then gosub 1000
600   rem
610   rem Backward pass
620   let dummy = loss.backward()
630   rem
640   rem Update parameters using gradient descent
650   let w_new = w - lr * w.grad
660   let b_new = b - lr * b.grad
670   rem
680   rem Detach and update (break computation graph)
690   let w = detach(w_new)
700   let b = detach(b_new)
710   rem
720   rem Zero gradients
730   let dummy2 = w.zero_grad()
740   let dummy3 = b.zero_grad()
750   rem
760 next epoch
770 rem
780 print ""
790 print "Training complete!"
800 print ""
810 print "Final weights:"
820 print w
830 print "Final bias:"
840 print b
850 print ""
860 print "Testing predictions:"
870 let test_logits = x_data @ w + b
880 print "Predictions:"
890 print test_logits
900 print "Expected: [[0],[0],[0],[1]] (approximately)"
910 rem
920 end
930 rem
940 rem Subroutine to print progress
1000 print "Epoch"
1010 print epoch
1020 print "Loss:"
1030 print loss
1040 return
