10 rem Subroutines with GOSUB/RETURN
20 rem
30 print "Main program start"
40 rem
50 let value = 5
60 print "Calling subroutine with value:"
70 print value
80 gosub 200
90 rem
100 let value = 10
110 print "Calling subroutine again with value:"
120 print value
130 gosub 200
140 rem
150 print "Main program end"
160 end
170 rem
200 rem Subroutine: square the value
210 print "In subroutine - squaring..."
220 let result = value * value
230 print "Result:"
240 print result
250 return
