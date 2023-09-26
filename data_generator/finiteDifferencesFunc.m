function deriv = finiteDifferencesFunc(X,method,dir,periodic)

n = length(X);

[centeredStencilLHS, decenteredStencilLHS, decenteredStencilRHS, centeredStencilRHS] = finiteDifferenceCoefficients(method);

LHS = makeEachMatrix(centeredStencilLHS,decenteredStencilLHS,n,periodic);
RHS = makeEachMatrix(centeredStencilRHS,decenteredStencilRHS,n,periodic);

LHS = sparse(LHS);
RHS = sparse(RHS);

% Apply metrics

LHSmetric = makeEachMatrix(centeredStencilLHS,decenteredStencilLHS,n,false);
RHSmetric = makeEachMatrix(centeredStencilRHS,decenteredStencilRHS,n,false);

dXdEta = LHSmetric\(RHSmetric*X');
LHS = bsxfun(@times,LHS,dXdEta');

if dir == 1 % For X direction
    deriv = @(U)(LHS\(RHS*U));
else % For Y direction
    
    % Transpose LHS and RHS
    LHS = LHS';
    RHS = RHS';
    
    deriv = @(U)((U*RHS)/LHS);
end


end
