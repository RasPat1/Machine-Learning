function [X, y] = generateData(N, samples)

X = zeros(samples, N);
y = zeros(samples, 1);

for i = 1:samples
	if (rand >= 0.5)
		X(i, :) = good(N);
		y(i) = 1;
	else
		X(i, :) = bad(N);
		y(i) = 0;
	end
end


end
