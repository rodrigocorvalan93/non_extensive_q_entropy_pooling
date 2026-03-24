function V=views_generator(X,Lower,Upper,ViewMethod)
  if ViewMethod=='random'
      ms=mean(std(X));
      V=zeros(size(X(:,Lower)));
      for k=1:size(X,1)
        V(k)=(1+2*abs(randn(1,1))*ms)*X(k,Lower) - X(k,Upper);
      endfor
      return
    elseif abs(ViewMethod)>0
      V=abs(ViewMethod)*X(:,min(Lower,Upper)) - X(:,max(Lower,Upper));
      return
    elseif ViewMethod~=0
      'warning no dio un ViewMethod adecuado se usara como si fuera 0'
    endif
    V=X(:,Lower) - X(:,Upper);
    return
endfunction