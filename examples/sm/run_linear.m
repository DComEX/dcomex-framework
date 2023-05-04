rng(123456);
Nd = 40;
alpha =  2;
beta  = -2;
sigma =  2;
data.x = linspace(1, 10, Nd);
data.y = alpha * data.x + beta + normrnd(0,sigma,1,Nd);
N = 100;
eps = 0.04;
d = 3;
sys_para.hard_bds = [ -5 -5  0;
		      5  5 10];
sys_para.conf = 0.68;
chi = chi2inv(sys_para.conf,d);
x = zeros(N,d);
f = zeros(N,1);
out = cell(N, 1);
x2 = zeros(N,d);
f2 = zeros(N, 1);
out2 = cell(N, 1);
for j = 1:d
  x(:, j) = random('Uniform', sys_para.hard_bds(1, j),sys_para.hard_bds(2, j), N,1);
end
for i = 1:N
  [f(i), out{i}] = loglike(x(i,:),data);
end
gen = 1;
p = 0;
End = false;
while 1
  old_p = p;
  plo = p;
  phi = 1.1;
  while phi - plo > 1e-12
    p = (plo + phi) / 2;
    temp = (p - old_p) * f;
    M1 = logsumexp(temp) - log(N);
    M2 = logsumexp(2 * temp) - log(N);
    if M2 - 2 * M1 > log(2); phi = p; else plo = p; end
  end
  if p >= 1
    p = 1;
    End = true;
  end
  weight = softmax((p - old_p)*f);
  x0 = x - weight' * x;
  cum_weight = [0 cumsum(weight)'];
  ind = sort(arrayfun(@(x) find(cum_weight <= x, 1, 'last'), rand(1, N)));
  for i = 1:N
    j = ind(i);
    V = out{j}.V;
    D = out{j}.D;
    Dp = D/p;
    do_correction = false;
    for l = 1:d
      sc = sqrt(Dp(l)/chi);
      if ~inside(x(j, :)' - sc*V(:,l), sys_para.hard_bds) || ...
	 ~inside(x(j, :)' + sc*V(:,l), sys_para.hard_bds)
	do_correction = true;
	break;
      end
    end
    if do_correction
      Dp_new  = adapt(x(j, :), Dp, V, sys_para);
      SIG =  V * diag(Dp_new) * V';
    else
      SIG = out{j}.inv_G/p;
    end
    bds = sys_para.hard_bds;
    xc = mvnrnd(x(j,:)' + 0.5*eps*SIG*p*out{j}.gradient(:), eps*SIG );
    if ~(any(xc < bds(1,:)) || any(xc > bds(2,:)))
      [lnfc_f,outc_f] = loglike(xc,data);
      gradientc = p * outc_f.gradient(:);
      tmpc = xc(:) - x(j,:)' - 0.5*eps*SIG*p*out{j}.gradient(:);
      tmpo = x(j,:)' - xc(:) - 0.5*eps*SIG*gradientc(:);
      qc = -(tmpc'*(eps*SIG\tmpc))/2;
      qo = -(tmpo'*(eps*SIG\tmpo))/2;
      r = exp(old_p*(lnfc_f - f(j))  + qo - qc);
      if 1 < r || rand() < r
	x(j, :) = xc;
	f(j) = lnfc_f;
	out{j} = outc_f;
      end
    end
    x2(i, :) = x(j, :);
    f2(i) = f(j);
    out2{i} = out{j};
  end
  gen = gen + 1;
  if End; break; end
  [x2, x, f2, f, out2, out] = deal(x, x2, f, f2, out, out2);
end
disp(mean(x2));
function [Dp] = adapt( x, Dp, V, sys_para)
  x = x(:);
  df = length(x);
  chi2 = chi2inv(sys_para.conf,df);
  bds = sys_para.hard_bds;
  width = bds(2,:)-bds(1,:);
  bds(2,:) = bds(2,:)+width;
  bds(1,:) = bds(1,:)-width;
  for l = 1:length(x)
    sc = sqrt(Dp(l)*chi2);
    pp = x + sc*V(:,l);
    flag = pp' <= bds(1,:);
    if any(flag > 0)
      c = abs(1./V(flag,l) .* (bds(1,flag)'-x(flag)));
      sc = min(c(:));
    end
    pp = x + sc*V(:,l);
    flag = pp' >= bds(2,:);
    if any(flag > 0)
      c = abs(1./V(flag,l) .* (bds(2,flag)'-x(flag)));
      sc = min(c(:));
    end
    pm = x - sc*V(:,l);
    flag = pm' <= bds(1,:);
    if any(flag > 0)
      c = abs(1./V(flag,l) .* (bds(1,flag)'-x(flag)));
      sc = min(c(:));
    end
    pm = x - sc*V(:,l);
    flag = pm' >= bds(2,:);
    if any(flag > 0)
      c = abs(1./V(flag,l) .* (bds(2,flag)'-x(flag)));
      sc = min(c(:));
    end
    Dp(l) = sc^2/chi2;
  end
end
function flag = inside( x, Box )
  x = x(:)';
  flag = ~any( x<Box(1,:) | x>Box(2,:) );
end
