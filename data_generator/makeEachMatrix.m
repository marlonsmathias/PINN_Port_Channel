function M = makeEachMatrix(C,D,n,periodic)
    % This builds either LHS or RHS matrices based on the finite differences coefficients
    M = diag(C(1)*ones(n,1));
    lc = length(C);
    [nd,ld] = size(D);

    if C(1) == 0 % If the RHS centered stencil has center 0, its left side should have the signal inverted as this is the stencil for derivatives, otherwise, its a stencil for filters
        invertStencil = -1;
    else
        invertStencil = 1;
    end
    
    for i = 1:lc-1
        inds = sub2ind([n n], 1:n, 1+mod((1:n)-1+i,n));
        M(inds) = C(i+1);
        
        inds = sub2ind([n n], 1:n, 1+mod((1:n)-1-i,n));
        M(inds) = invertStencil*C(i+1);
    end
    
    if ~periodic
        M(1:nd,:) = 0;
        M(end:-1:end-nd+1,:) = 0;

        M(1:nd,1:ld) = D;
        M(end:-1:end-nd+1,end:-1:end-ld+1) = invertStencil*D;
    end

end