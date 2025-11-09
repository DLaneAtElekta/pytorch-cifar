10 rem Gradient Basics - Simple Autograd Example
20 rem This demonstrates the core gradient operations
30 rem in tensorBASIC using PyTorch autograd
40 rem
50 print "TensorBASIC Gradient Operations Demo"
60 print "===================================="
70 print ""
80 rem
90 rem Example 1: Simple gradient computation
100 print "Example 1: Computing gradients"
110 print "-------------------------------"
120 rem Create a parameter: x = 2.0 (requires gradient)
130 let x = param(1)
140 let x[0] = 2.0
150 print "x ="
160 print x
170 rem
180 rem Compute y = x^2
190 let y = x * x
200 print "y = x * x ="
210 print y
220 rem
230 rem Compute gradient dy/dx
240 let dummy = y.backward()
250 print "Gradient dy/dx ="
260 print x.grad
270 print "Expected: 2*x = 4.0"
280 print ""
290 rem
300 rem Example 2: Training a single parameter
310 print "Example 2: Training loop"
320 print "------------------------"
330 rem Goal: minimize f(w) = (w - 3)^2
340 rem Starting from w = 0, should converge to w ≈ 3
350 let w = param(1)
360 let w[0] = 0.0
370 let lr = 0.1
380 rem
390 print "Minimizing f(w) = (w - 3)^2"
400 print "Starting w = 0.0"
410 print ""
420 rem
430 for iter = 1 to 10 step 1
440   rem Compute loss
450   let diff = w - 3.0
460   let loss = diff * diff
470   rem
480   rem Backward pass
490   let dummy2 = loss.backward()
500   rem
510   rem Update
520   let w_new = w - lr * w.grad
530   let w = detach(w_new)
540   rem
550   rem Zero gradient
560   let dummy3 = w.zero_grad()
570   rem
580   rem Print progress
590   if iter <= 5 then gosub 1000
600 next iter
610 rem
620 print "..."
630 print "Final w ="
640 print w
650 print "Expected: w ≈ 3.0"
660 print ""
670 rem
680 print "Demo complete!"
690 end
700 rem
710 rem Print subroutine
1000 print "Iteration"
1010 print iter
1020 print "w ="
1030 print w
1040 print "loss ="
1050 print loss
1060 return
