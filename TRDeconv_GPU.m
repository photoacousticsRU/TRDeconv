%Deconvolutional filtering of 3D optical-acoustic data using Tikhonov regularization.


function out_deconv = TRDeconv_GPU(inp1, inp2, inp3, inp4, inp5, inp6)

%inp1 - 3D array of data
%inp2, inp3 - x and y position of 'sample' scan
%inp4, inp5 - z-begin, z-end of 'sample'
%inp6 - peak size of 'sample' scan. Used for noise clipping.

tic
%Get 'sample' A-scan

Ascan = squeeze(inp1(inp2, inp3, inp4:inp5));
Ascan_hilb=abs(hilbert(Ascan))/max(abs(hilbert(Ascan)));

[pcs,lcs] = max(Ascan_hilb); %detect peak

delta = inp6  

Ascan_T = Ascan;
Ascan_T(1:lcs-delta)=0; % zeroing values around peak
Ascan_T(lcs+delta:end)=0;

%Detect peak width
SC0=Ascan_hilb(lcs-30:lcs+30);
NSC0=0;

for i=1:1:61
    
    if  SC0(i)-0.5<0
        NSC0=NSC0;
        else
        NSC0=NSC0+1;                                 
    end    
end
                      
N0=NSC0 %count of values higher than 0.5*Max

%Build ideal A-scan (IR)

IR = Ascan.*0; 
IR(lcs)=1;
IR(lcs+1)=-1;

%Solving deconvolution params moving towards ideal A-scan (IR)

NF =3;       %length of filter kernel
bet_a1=0.01; %intervals of regularization
bet_b1=0.31;
nk=0;        %iteration counter

%Running solver by halfing regularization params
for k=1:1:10; 
    nk=nk+1;
    l=abs(bet_b1-bet_a1);
    bet(k)=bet_a1 +l/2;
    bet1=bet_a1+l/4;
    bet2=bet_b1-l/4;

    FdeconvNew = fun_RegOAscan(Ascan_T,IR,NF,bet(k));
    Ascan_deconv = fun_Check_signal_2((Ascan)',FdeconvNew);
    SC1=abs(hilbert(Ascan_deconv(lcs-30:lcs+30)))/max(abs(hilbert(Ascan_deconv(lcs-30:lcs+30))));

    FdeconvNew1 = fun_RegOAscan(Ascan_T,IR,NF,bet1);
    Ascan_deconv1 = fun_Check_signal_2((Ascan)',FdeconvNew1);
    SC1_a=abs(hilbert(Ascan_deconv1(lcs-30:lcs+30)))/max(abs(hilbert(Ascan_deconv1(lcs-30:lcs+30))));

    FdeconvNew2 = fun_RegOAscan(Ascan_T,IR,NF,bet2);
    Ascan_deconv2 = fun_Check_signal_2((Ascan)',FdeconvNew2);
    SC1_b=abs(hilbert(Ascan_deconv2(lcs-30:lcs+30)))/max(abs(hilbert(Ascan_deconv2(lcs-30:lcs+30))));

    %Reducing scan width
    NSC1=0; NSC1_a=0; NSC1_b=0; 

    for i=1:1:61
    
        if  SC1(i)-0.5<0
            NSC1=NSC1;
            else
           NSC1=NSC1+1;
        end
         if  SC1_a(i)-0.5<0
             NSC1_a=NSC1_a;
            else
             NSC1_a=NSC1_a+1;
        end
         if  SC1_b(i)-0.5<0
            NSC1_b=NSC1_b;
            else
            NSC1_b=NSC1_b+1;
        end
    end
                 
    N1=NSC1
    N2=NSC1_a
    N3=NSC1_b

   %check width peaks to finish solver

    if N2<=N1
       bet_a1=bet_a1; bet_b1=bet(k); bet(k+1)=bet1;
       else
        bet_b1=bet_b1; bet_a1=bet(k); bet(k+1)=bet2;
    end
    if abs(N3-N2) < 1;
     break
    end
end
 
bet_f=bet(k);  %optimal regularization value
 
 compr=100*(N0-N1)/N0; %compression rate
 
 %Save filter params and stats

save('TRD_var.mat','FdeconvNew', 'bet_f', 'compr', 'nk');
 
% Final applying filter

Fdeconv3d =reshape(FdeconvNew, 1, 1, 3);
Fdeconv3d_gpu = gpuArray(Fdeconv3d); %Move kernel to gpu
inp1_gpu = gpuArray(inp1);
out_deconv = convn(inp1_gpu,Fdeconv3d_gpu, 'SAME');
 %out_deconv = fun_Check_signal_2_3D(inp1,FdeconvNew) ;
 toc
end