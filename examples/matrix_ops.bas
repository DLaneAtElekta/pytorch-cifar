10 rem Matrix operations in tensorBASIC
20 rem Create two 3x3 matrices
30 dim a(3,3)
40 dim b(3,3)
50 dim c(3,3)
60 rem
70 rem Initialize matrix a
80 let a = [[1,2,3],[4,5,6],[7,8,9]]
90 rem
100 rem Initialize matrix b
110 let b = [[9,8,7],[6,5,4],[3,2,1]]
120 rem
130 print "Matrix A:"
140 print a
150 rem
160 print "Matrix B:"
170 print b
180 rem
190 rem Matrix multiplication
200 let c = a @ b
210 rem
220 print "A @ B (matrix multiply):"
230 print c
240 rem
250 rem Element-wise operations
260 let sum = a + b
270 print "A + B:"
280 print sum
290 rem
300 rem Tensor slicing
310 let slice = a[0:2, 1:3]
320 print "A[0:2, 1:3] (slice):"
330 print slice
340 rem
350 end
