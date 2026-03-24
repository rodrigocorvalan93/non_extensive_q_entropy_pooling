function [p_ Lx lv]= EntropyProg(p,A = [],b=[],Aeq,beq,entropy_family='S',q=1,g=[],dg=[])
% ADVERTENCIA 1: si la entropy_family es Shannon no corresponden q ni g (y no se usan, haya lo que haya)
% si la entropy_family es Tsallis debe haber q y no debe ser 1 (si q=1 da error), y g=@(x)x+1 (se usa eso, haya lo que haya)
% si la entropy_family es Renyi debe haber q y no debe ser 1 (si q=1 da error), y g=@(x)ln(x) (se usa eso, haya lo que haya)
% si la entropy_family es General debe haber q y no debe ser 1 (si q=1 da error), y debe darse g=@(x)x+1 (si no da error)

if entropy_family~='T' && entropy_family~='R' && entropy_family~='S'
  display("Warning, se toma entropy_family='G'");
endif
if entropy_family~='S'
  if q==1
    error('q debe ser distinto de 1')
  endif
  if entropy_family=='T'
    g=@(x)x-1;
    dg=@(x)1;
  elseif entropy_family=='R'
    g=@(x)log(x);
    dg=@(x)1./x;  
  endif
  if entropy_family=='G'
    if isempty(g) || isempty(dg)
      error('con entropy_family G hay que dar g y dg')
    endif
  endif
endif

if entropy_family=='S'
  fe=@(x)x'*(log(x)-log(p));
else
  fe=@(x)(1/(q-1))*g(((x./p).^q)'*p);
endif
K_=size(A,1);
K=size(Aeq,1);

if q<1 %Admite resolucion via busqueda de ceros del Lagrangiano por Newton-Raphson para longitud de p<10000
  Lx=@(X)fe(X(1:length(p)))+X(length(p)+1:length(p)+K_)'*(A*X(1:length(p))-b)+X(length(p)+K_+1:end)'*(Aeq*X(1:length(p))-beq);
  GRADL=@(X)JACOB_APROX(Lx,X)';
  PC=NR_MULTI(GRADL,[p;zeros(K_+K,1)],17);
  tol=10^-8;
  contador=1;
  while GRADL(PC)>tol && contador<4097
    contador++;
    PC=NR_MULTI(GRADL,[p;10*rand(K_,1);10*randn(K,1)],17);
  endwhile
  if GRADL(PC)<tol
    p_=PC(1:length(p));
    lv=[PC(length(p)+1:length(p)+K_);PC(length(p)+K_+1:end)];
    return
    else
    error('no se encontro solucion')
  endif
endif

A_=A';
b_=b';
Aeq_=Aeq';
beq_=beq';
x0=zeros(K_+K,1);
%if entropy_family=='S'
%  x0=zeros(K_+K,1);
%elseif entropy_family=='T'
%    [p_S LxS lvS]=EntropyProg(p,A,b,Aeq,beq)
%    x0=lvS    
%  else
%    %[p_S LxS lvS]= EntropyProg(p,A,b,Aeq,beq);
%    [p_T LxT lvT]=EntropyProg(p,A,b,Aeq,beq,'T',q);
%    x0=lvT;
%    %x0=lvS;
%endif
InqMat=-eye(K_+K); InqMat(K_+1:end,:)=[];
InqVec=zeros(K_,1);

if K_+K==0
  p=p_;
  display('warning, no hay views');
  return
endif
  
options = optimset('GradObj','on','Hessian','on'); %quizas lo dejo en off de modo que usara metodos sin Hessiano

if ~K_
    v=fminunc(@nestedfunU,x0,options);
    A_=zeros(size(x));
    b_=0;
    l=0;
    p_=equis(p,A_,l,Aeq_,v,entropy_family,q);
    Lx=@(x)fe(x)+(x'*A_-b_)*l+(x'*Aeq_-beq_)*v;
else
    lv=fmincon(@nestedfunC,x0,InqMat,InqVec,[],[],[],[],[],options);
    l=lv(1:K_);
    v=lv(K_+1:end);
    p_=equis(p,A_,l,Aeq_,v,entropy_family,q);
    Lx=@(x)fe(x)+(x'*A_-b_)*l+(x'*Aeq_-beq_)*v;
    GLT=@(X)[(q/(q-1))*((X(1:length(p))./p)).^(q-1)+Aeq'*v+A'*l;Aeq*X(1:length(p))-beq;A*X(1:length(p))-b];
    GLT(p_);
endif
    
    function [mL g H] = nestedfunU(v)
        %'unconstrained'
        A_=zeros(size(x));
        b_=0;
        l=0;
        x=equis(p,A_,l,Aeq_,v,entropy_family,q);
        x=max(x,10^(-32));
        %L=x'*(log(x)-log(p)+Aeq_*v)-beq_*v;
        %L=x'*(log(x)-log(p))+(x'*A_-b_)*l+(x'*Aeq_-beq_)*v; %reescribo L para unificarla para ambas situaciones const y unconst
        L=fe(x)+(x'*A_-b_)*l+(x'*Aeq_-beq_)*v; %reescribo L para unificarla para ambas situaciones const y unconst
        
        Lx=@(x)fe(x)+(x'*A_-b_)*l+(x'*Aeq_-beq_)*v; %Por si quiero testear a mano el minimo
        mL=-L;    
        
        g = [beq-Aeq*x];    
        H = [Aeq*((x*ones(1,K)).*Aeq_)];  % Hessian computed by Chen Qing, Lin Daimin, Meng Yanyan, Wang Weijun 
    endfunction

    function [mL g H] = nestedfunC(lv)
        %'constrained'
        l=lv(1:K_);
        v=lv(K_+1:end);
        x=equis(p,A_,l,Aeq_,v,entropy_family,q);
        %pause
        x=max(x,10^(-32));
        L=fe(x)+(x'*A_-b_)*l+(x'*Aeq_-beq_)*v;  %reescribo L para unificarla para ambas situaciones const y unconst
        Lx=@(x)fe(x)+(x'*A_-b_)*l+(x'*Aeq_-beq_)*v; %Por si quiero testear a mano el minimo
        mL=-L;
    
        g = [b-A*x; beq-Aeq*x];    
        H = [A*((x*ones(1,K_)).*A_)  A*((x*ones(1,K)).*Aeq_) % Hessian computed by Chen Qing, Lin Daimin, Meng Yanyan, Wang Weijun 
            Aeq*((x*ones(1,K_)).*A_)   Aeq*((x*ones(1,K)).*Aeq_)];  
        %[mL g H]    
    endfunction
    
    function x=equis(p,A_,l,Aeq_,v,entropy_family,q)
        if entropy_family=='S'
          x=exp(log(p)-1-A_*l-Aeq_*v);
        elseif entropy_family=='T'
           x=((1/q-1)*(A_*l+Aeq_*v)).^(1/(q-1)).*p;
        elseif entropy_family=='R'
         [xini Lx lv]=EntropyProg(p,A,b,Aeq,beq,'T',q);  
         %xini
         F_PUNTOFIJO=@(xini)((1/q-1)*(((xini./p).^q)'*p)*(A'*lv(1)+Aeq'*lv(2:3))).^(1/(q-1)).*p;
         %F_PUNTOFIJO(xini)
         %pause
         
         xalter=p;
         x=despeja_itera_x(xini,xalter,F_PUNTOFIJO);
         % empieza la busqueda de ceros usando la solucion con Tsallis y el mismo q porque x+1 es la aprox de orden 1 de ln(x)
         % si no converge desde al solucion de Tsallis se prueba desde el prior p
         % Renyi funcionaria tambien bajo el caso general pero preferimos exhibir el despeje explicitamente
         % borrar x=despeja_x(@(x)(1/(q-1))*(1./(((x./p).^q)'*p))*(x./p).^(q-1)+A_*l+Aeq_*v , ((1/q-1)*(A_*l+Aeq_*v)).^(1/(q-1)).*p);
        else 
         % x=despeja_x(@(x)(1/(q-1))*dg((x./p).^q)'*p))*(x./p).^(q-1)+A_*l+Aeq_*v , ((1/q-1)*(A_*l+Aeq_*v)).^(1/(q-1)).*p);
         xini=((1/q-1)*(A_*l+Aeq_*v).^(1/(q-1))).*p;
         xalter=p;  
         F_PUNTOFIJO=@(x)((1/q-1)*(1/dg(((x./p).^q)'*p))*(A_*l+Aeq_*v)).^(1/(q-1)).*p;
         x=despeja_itera_x(xini,xalter,F_PUNTOFIJO);
        endif
    endfunction
    
    function x=despeja_itera_x(xini,xalter,F_PUNTOFIJO)
        tol=10^-8;
        flag=0;
        x=abs(xini)/sum(abs(xini)); % Se busca x en el simplex sum(x)==1, x_j>=0 para todo j; es lo que hace U (abajo)
        %' Real?'
        %isreal((1/q-1)*(((x./p).^q)'*p)*(A_*l+Aeq_*v))
        %x
        %F_PUNTOFIJO(x)
        %pause
        F_PUNTOFIJO=@(x)F_PUNTOFIJO(x)*norm(x,1)/norm(F_PUNTOFIJO(x),1);
        %'MENSAJE'
        %norm(x-F_PUNTOFIJO(x))
        U=@(x)abs(F_PUNTOFIJO(x))/sum(abs(F_PUNTOFIJO(x))); % Manda el simplex en si mismo. Hay punto fijo por teorema de Brouwer 
        iterador=0;
        alfa=0.5^iterador;
        G=@(x)alfa*U(x)+(1-alfa)*x; % Para que converja buscamos una contraccion con el mismo punto fijo (iteracion de Mann)
        while iterador<17 && norm(x-F_PUNTOFIJO(x))>tol
          contador=1;
          while norm(x-F_PUNTOFIJO(x))>tol && contador<64
            x=G(x);
            x=abs(x)/sum(abs(x));
            contador++;
          endwhile
          iterador++;
          alfa=0.5^iterador;
          G=@(x)alfa*U(x)+(1-alfa)*x;
          if norm(x-F_PUNTOFIJO(x))>norm(xini-F_PUNTOFIJO(xini))
            x=xini;
          endif
        endwhile
        if norm(x-F_PUNTOFIJO(x))<tol
          %'salida1'
          return
        endif        
        %contador
        if norm(x-F_PUNTOFIJO(x))>tol
          flag=1
          x=abs(xalter)/sum(abs(xalter)); %si no converge desde la sol de Tsallis, pruebo a empezar desde el prior
          iterador=0;
          alfa=0.5^iterador;
          G=@(x)alfa*U(x)+(1-alfa)*x;
          while iterador<17 && norm(x-F_PUNTOFIJO(x))>tol
            contador=1;
            while norm(x-F_PUNTOFIJO(x))>tol && contador<64
              x=G(x);
              x=abs(x)/sum(abs(x));
              contador++;
            endwhile
            iterador++;
            alfa=0.5^iterador;
            G=@(x)alfa*U(x)+(1-alfa)*x;
            if norm(x-F_PUNTOFIJO(x))>norm(xalter-F_PUNTOFIJO(xalter))
              x=xalter;
            endif 
          endwhile         
        endif
        if norm(x-F_PUNTOFIJO(x))<tol
          %'salida2'
          return
        endif
        while flag<1024 && norm(x-F_PUNTOFIJO(x))>tol
          %'flag'
          ++flag;
          iterador=0;
          alfa=0.5^iterador;
          G=@(x)alfa*U(x)+(1-alfa)*x;
          xseed=rand(size(xini));
          xseed=xseed/sum(xseed);
          x=xseed;
          while iterador<17 && norm(x-F_PUNTOFIJO(x))>tol
            contador=1;
            while norm(x-F_PUNTOFIJO(x))>tol && contador<64
              x=G(x);
              x=abs(x)/sum(abs(x));
              contador++;
            endwhile
            iterador++;
            alfa=0.5^iterador;
            G=@(x)alfa*U(x)+(1-alfa)*x;
            if norm(x-F_PUNTOFIJO(x))>norm(xseed-F_PUNTOFIJO(xseed))
              x=xseed;
            endif 
          endwhile
        endwhile
        if norm(x-F_PUNTOFIJO(x))<tol
          %'salida3'
          return
        endif
        if flag>=1024 && norm(x-F_PUNTOFIJO(x))>tol
          error('no se pudo hallar punto fijo')  
        endif
    endfunction
    
    function xsol=NR_MULTI(H,p1,N)
      DH=@(X)JACOB_APROX(H,X);
      xsol=Newt_Raph(H,DH,p1,N);
    endfunction

    function Xsol=Newt_Raph(h,dh,p1,N)
      Xsol=p1;
      for k=1:N
        Xsol=Xsol-inv(dh(Xsol))*h(Xsol);
      endfor
    endfunction

    function J=JACOB_APROX(H,X,epsi)
      if nargin==2
        epsi=sqrt(sqrt(eps(1+norm(X))));
      endif
      I=eye(length(X));
      for k=1:length(X)
        J(:,k)=(H(X+epsi*I(:,k))-H(X-epsi*I(:,k)))/(2*epsi);
      endfor
    endfunction  
end  