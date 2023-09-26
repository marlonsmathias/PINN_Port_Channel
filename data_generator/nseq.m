function [Et,Ut,Vt] = nseq(E,U,V,pars,dx,dy)

    D = E + pars.H;
 
    UyVx = dy(U) + dx(V);

    Am = 0.5 * pars.C .* pars.deltaX .* pars.deltaY * sqrt(dx(U).^2 + UyVx.^2 + dy(V).^2);

    Fx = dx(2*pars.H.*Am.*dx(U)) + dy(pars.H.*Am.*UyVx);
    Fy = dy(2*pars.H.*Am.*dy(V)) + dx(pars.H.*Am.*UyVx);

    Et = -dx(U.*D) - dy(V.*D);

    % Acconting for convective effects
    Ut = 1./D.*(-U.*Et -dx(U.^2.*D) -dy(U.*V.*D) + Fx) -pars.g*dx(E);
    Vt = 1./D.*(-V.*Et -dy(V.^2.*D) -dx(U.*V.*D) + Fy) -pars.g*dy(E);

    % Disregarding convective effects
    %Ut = 1./D.*(-U.*Et + Fx) -pars.g*dx(E);
    %Vt = 1./D.*(-V.*Et + Fy) -pars.g*dy(E);

end