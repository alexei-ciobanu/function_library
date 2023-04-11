function [l] = laguerre(n, alpha)

i = 1:n;
a = (2*i-1) + alpha;
b = sqrt( i(1:n-1) .* ((1:n-1) + alpha) );
CM = diag(a) + diag(b,1) + diag(b,-1);

c = (-1)^n/factorial(n) * poly(CM);

l = polyval(c,x)