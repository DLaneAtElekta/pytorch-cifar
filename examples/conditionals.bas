10 rem Conditional statements in tensorBASIC
20 rem
30 let x = 10
40 let y = 20
50 rem
60 print "Testing conditionals:"
70 print "x = 10, y = 20"
80 rem
90 if x < y then print "x is less than y"
100 rem
110 if x > y then print "x is greater than y"
120 rem
130 if x == 10 then print "x equals 10"
140 rem
150 print "Using IF with GOTO:"
160 let counter = 0
170 rem
180 if counter >= 5 then 220
190 print counter
200 let counter = counter + 1
210 goto 180
220 rem
230 print "Counter reached 5"
240 end
