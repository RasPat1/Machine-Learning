function v = good(N)


v = 1:N;

for i= 1:N
	randIndex = floor(unifrnd(i, N + 1));
	tmp = v(i);
	v(i) = v(randIndex);
	v(randIndex) = tmp;
end

end