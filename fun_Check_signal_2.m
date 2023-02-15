function YY = fun_Check_signal_2(restored_2real1,Fdeconv)

r = [restored_2real1 zeros(1,length(Fdeconv)-1)];
c = [restored_2real1(1) zeros(1,length(Fdeconv)-1)];

xConv = toeplitz(c, r);

if mod(numel(Fdeconv),2) == 0
    n_crop = numel(Fdeconv)/2;
    xConv = xConv(:, n_crop+1:end-n_crop+1);
else
    n_crop = floor(numel(Fdeconv)/2);
    xConv = xConv(:, n_crop+1:end-n_crop);
end

YY = Fdeconv*xConv;
