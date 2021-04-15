


function [A_full, B_full] = L1_SCCA(x,z,penalty_x,penalty_z,n_can_var)
    x_col = size(x,2); %number of connections == q
    z_col = size(z,2); %number of behaviour variables == p

    if x_col<2 %throw error if less than 2 variables
        error('Need at least 2 columns (features) in dataset x'); 
    end 
    if z_col<2
        error('Need at least 2 columns (features) in dataset z'); 
    end

    x = zscore(x); %standardize matrices to have zero mean and unit variance
    z = zscore(z);

    % initialize v
    y = z'*real(sqrtm(x*x'));
    [v_init,~,~] = svd(y); %[pxp] 
    v_init = v_init(:,1:n_can_var); %[pxp] matrix of px1 v initial vectors 

    % iterate for each canonical variable
    xres = x; % keeps track of x [[nxn]...]
    zres = z; % keeps track of z [[nxp]...]

    u=zeros(x_col,z_col); % keeps track of u vector [[qx1]...] for each retained variable
    v=zeros(z_col,n_can_var); % keeps track of v vector [[px1]...] for each retained variable
    d=zeros(1,n_can_var); % keeps track of d scalar [[0x0]...] for each retained variable

    for k = 1:n_can_var % for each retained variable
        [cur_d,cur_u,cur_v] = SparseCCA(xres, zres, v_init(:,k), penalty_x, penalty_z); %SCCA solves for d,u,v
        d(k) = cur_d; 
        xres = [xres; sqrt(cur_d)*cur_u']; % recalculate x matrix using d and u
        zres = [zres;-sqrt(cur_d)*cur_v']; % recalculate z matrix using d and v
        u(:,k) = cur_u; 
        v(:,k) = cur_v;
    end

%     A_full = zeros(x_col,n_can_var); %initialize coefficient matrix to project x matrix
%     A_full(sum(x.^2,1)~=0,:) = u; %excludes where sum of squared columns in x is 0  (variable data not present)
    A_full = u;

%     B_full = zeros(z_col,n_can_var); %initialize coefficient matrix to project z matrix
%     B_full(sum(z.^2,1)~=0,:) = v; %excludes where sum of squared rows in z is 0  (variable data not present)
    B_full = v;

    U_proj = x*u; %projection of x matrix
    V_proj = z*v; %projection of z matrix
    corrs = zeros(1,n_can_var); %initialize correlation values between each canonical variable

    for i=1:n_can_var
        corrs(i) = corr(U_proj(:,i),V_proj(:,i)); %determine correlation between each column in the projected matrices
    end
end

function [d,u,v] = SparseCCA(x, z, v, penalty_x, penalty_z)

niter = 15;

x_col = size(x,2); %number of connections == q
z_col = size(z,2); %number of behaviour variables == p
v_len = length(v); %number of retained variables == p

v_old = randn(1,v_len); %initialize v vector [1xp]
u = randn(x_col,1); %initialize u vector [qx1]
i = 1;

    while i < niter && sum(abs(v_old(:)-v(:)))>1e-06
        if(any(isnan(u)) || any(isnan(v))) %if vectors contain NaN, set previous v vector to zero vector
            v = zeros(1,v_len);
            v_old = v;
        end

        %Update U
        argu = x'*(z*v); %coefficients for connections matrix [qx1]
        delU = BinarySearch(argu,penalty_x*sqrt(x_col)); %binary search for delta such that norm(u,1) <= penalty for x;
        Su = S(argu,delU); 
        u = Su/L2norm(Su); %determine coefficients for x matrix

        %Update V
        v_old = v; %keep previously calculated v
        argv = (u'*x')*z; %coefficients for behaviour matrix [1xp]
        delV = BinarySearch(argv,penalty_z*sqrt(z_col)); %binary search for delta such that norm(v,1) <= penalty for z;
        sv = S(argv,delV); 
        v = (sv/L2norm(sv))'; %determine coefficients for z matrix

        i = i+1;
    end
    
d = (x*u)'*(z*v);

    if(any(isnan(u)) || any(isnan(v))) %if NaN is within u or v, throw warning
        u = zeros(x_col,1);
        v = zeros(z_col,1);
        d = 0;
        warning('CCA failed')
    end
end

function delta = BinarySearch(argu,sumabs) 
% argu: coefficients for matrix [qx1]
% sumabs: penalty for matrix*sqrt(number of connections) a.k.a. c*sqrt(n or p)
    if norm(argu)==0 || sum(abs(argu/L2norm(argu)),'all')<=sumabs %if norm(coefficents) == 0 or norm(coefficients) < constraint for vector, then set delta to 0
        delta = 0;
    else % otherwise use binary search to determine delta
        del1 = 0; % minimum value for delta
        del2 = max(abs(argu)) - 1e-05; % maximum value for delta
        iter=1; 

        while iter<150 
            su = S(argu,(del1+del2)/2); % soft thresholding operator on coefficients for u, with threshold == midpoint delta
            if(sum(abs(su/L2norm(su)))<sumabs) % if u < constraint, set max delta to midpoint
                del2 = (del1+del2)/2;
            else % otherwise, set min delta to midpoint
                del1 = (del1+del2)/2;
            end

            if((del2-del1)<1e-06) % if max and min have converged, set delta to midpoint
                delta = (del1+del2)/2;
                return
            end
            iter = iter+1;
        end
        warning('Did not converge') % if iterations exceed 150, stop searching and give warning
        delta = (del1+del2)/2;      
    end 
end

function u = u_fun(M,s,del)
    u = wthresh(M,s,del)/norm(wthresh(M,s,del)); 
end    

function a = L2norm(vec)
a = norm(vec);
    if(a==0)
        a = 0.05;
    end
end

function res = S(x,d)
res = sign(x).*max(0,abs(x)-d);
end