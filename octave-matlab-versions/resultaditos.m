CON VIEWMETHOD 1.4
q=1.1
>> norm(p-p_)/norm(p)
ans = 0.046030
>> norm(p-p_Clasico)/norm(p)
ans = 0.046042 %el clasico siempre es el mismo
%%%%%%%%%%%%%%%%%%%%%
q=1.3
>> norm(p-p_)/norm(p)
ans = 0.045991
%%%%%%%%%%%%%%%%%%%%%
q=1.5
>> norm(p-p_)/norm(p)
ans = 0.045968
%%%%%%%%%%%%%%%%%%%%%
q=1.7
>> norm(p-p_)/norm(p)
ans = 0.045953
%%%%%%%%%%%%%%%%%%%%%
q=1.9
>> norm(p-p_)/norm(p)
ans = 0.045945
%%%%%%%%%%%%%%%%%%%%%
q=2 %Este es el mejor, a partir de aqui se empieza a deteriorar
>> norm(p-p_)/norm(p)
ans = 0.045944
%%%%%%%%%%%%%%%%%%%%%
q=2.1
>> norm(p-p_)/norm(p)
ans = 0.045945
%%%%%%%%%%%%%%%%%%%%%
q=2.3
>> norm(p-p_)/norm(p)
ans = 0.045953
%%%%%%%%%%%%%%%%%%%%%
q=2.5
>> norm(p-p_)/norm(p)
ans = 0.045969
