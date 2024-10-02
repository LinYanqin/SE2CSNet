clc;clear all;
load('predict/pre.mat');

final = pre(1, :);
max = max(final);
min = min(final);
mean = (max + min) * 0.5;
for i = 1:length(final)
    if final(i) >= mean
        final(i) = 1;
    elseif final(i) < mean
        final(i) = 0;
    end
end

figure;
subplot(211);
plot(x(:, 1));
subplot(212);
stem(final, 'Marker', 'none');
                                        