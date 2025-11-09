10 rem Test checkpoint/resume functionality
20 print "Starting computation..."
30 let total = 0
40 for i = 1 to 100 step 1
50   let total = total + i
60   print total
70 next i
80 print "Final total:"
90 print total
100 end
