10 rem Loop examples in tensorBASIC
20 rem This demonstrates checkpoint resumption
30 rem
40 print "Counting with FOR loop:"
50 rem
60 for i = 1 to 10 step 1
70   print i
80 next i
90 rem
100 print "Computing factorial of 5:"
110 let n = 5
120 let factorial = 1
130 for i = 1 to n step 1
140   let factorial = factorial * i
150 next i
160 print factorial
170 rem
180 print "Done!"
190 end
