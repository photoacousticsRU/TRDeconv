function Fdeconv1 = fun_RegOAscan(Vec1,Vec2,F,bet)
N = length(Vec2);
MatrixC1= circulant(Vec1,1); 
MatrixC=MatrixC1(:,1:F);
I=eye(F);
%I=I1(1:F,:);
MatrixG=I .* bet;

Fdeconv_prep=inv(MatrixC' * MatrixC + MatrixG' * MatrixG) * MatrixC' ;
 
for p=1:F
    Fdeconv1(p)=0;
end

 for s=1:F
  for ss=1:N
          Fdeconv1(s)=Fdeconv1(s)+Fdeconv_prep(s,ss)*Vec2(ss); 
  end
 end

end