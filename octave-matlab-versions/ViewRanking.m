function p_ = ViewRanking(X,p,Lower,Upper,ViewMethod,entropy_family='S',q=1)

[J,N]=size(X);
K=length(Lower);

% constrain probabilities to sum to one...
Aeq = ones(1,J);  
beq=1;
% ...constrain the expectations...
V=views_generator(X,Lower,Upper,ViewMethod);

A = V';
b = 0;

% ...compute posterior probabilities

% ...compute posterior probabilities
%p_ = EntropyProg(p,A,b,Aeq,beq);  
p_ = EntropyProg(p,A,b,Aeq,beq,entropy_family,q);
endfunction