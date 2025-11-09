10 rem Linear Regression with Gradient Descent
20 rem This example trains a simple linear model y = w*x + b
30 rem using tensorBASIC gradient operations (PyTorch autograd)
40 rem
50 print "Linear Regression Training Example"
60 print "==================================="
70 print ""
80 rem
90 rem Create training data: y = 2*x + 3 + noise
100 rem Using 10 data points
110 dim x_data(10, 1)
120 let x_data = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
130 dim y_data(10, 1)
140 let y_data = [[5],[7],[9],[11],[13],[15],[17],[19],[21],[23]]
150 rem
160 print "Training data loaded (10 samples)"
170 print ""
180 rem
190 rem Initialize model parameters with gradient tracking
200 rem w: weight (1x1), b: bias (1x1)
210 let w = param(1, 1)
220 let b = param(1, 1)
230 rem
240 print "Initial parameters:"
250 print "w ="
260 print w
270 print "b ="
280 print b
290 print ""
300 rem
310 rem Hyperparameters
320 let lr = 0.01
330 let epochs = 100
340 rem
350 print "Training with learning rate 0.01 for 100 epochs..."
360 print ""
370 rem
380 rem Training loop
390 for epoch = 1 to epochs step 1
400   rem Forward pass: y_pred = x @ w + b
410   let y_pred = x_data @ w + b
420   rem
430   rem Compute mean squared error loss
440   let diff = y_pred - y_data
450   let squared = diff * diff
460   let loss = squared.mean()
470   rem
480   rem Print loss every 10 epochs
490   let check = epoch - 10 * (epoch / 10)
500   if check == 0 then gosub 1000
510   rem
520   rem Backward pass - compute gradients
530   let dummy = loss.backward()
540   rem
550   rem Manual gradient descent update (no optimizer needed)
560   rem w = w - lr * w.grad (detach from computation graph)
570   let w_grad = w.grad
580   let w_update = lr * w_grad
590   let w_new = w - w_update
600   let w = detach(w_new)
610   rem
620   rem b = b - lr * b.grad (detach from computation graph)
630   let b_grad = b.grad
640   let b_update = lr * b_grad
650   let b_new = b - b_update
660   let b = detach(b_new)
690   rem
700   rem Zero gradients for next iteration
710   let dummy2 = w.zero_grad()
720   let dummy3 = b.zero_grad()
730   rem
740 next epoch
750 rem
760 print ""
770 print "Training complete!"
780 print ""
790 print "Final parameters:"
800 print "w ="
810 print w
820 print "b ="
830 print b
840 print ""
850 print "Expected: w ≈ 2.0, b ≈ 3.0"
860 rem
870 end
880 rem
890 rem Subroutine to print loss
1000 print "Epoch"
1010 print epoch
1020 print "Loss:"
1030 print loss
1040 return
