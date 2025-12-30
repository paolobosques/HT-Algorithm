function [X] = sylvesterf(A,B,C)

X = sylvester(full(A),full(B),full(C));

end