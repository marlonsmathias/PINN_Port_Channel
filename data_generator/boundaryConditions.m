function [E,U,V] = boundaryConditions(t,E,U,V)

    % Dirichlet condition for velocities at the canal sides
    U(:,[1 end]) = 0;
    V(:,[1 end]) = 0;

    % Neumann condition for velocities at the canal inlet
    U(1,:) = 4/3*U(2,:) - 1/3*U(3,:);
    V(1,:) = 4/3*V(2,:) - 1/3*V(3,:);

    % Dirichlet condition for velocities at the canal end
    U(end,:) = 0;
    V(end,:) = 0;

    % Neumann condition for height at the sides
    E(:,1) = 4/3*E(:,2) - 1/3*E(:,3);
    E(:,end) = 4/3*E(:,end-1) - 1/3*E(:,end-2);

    % Neumann condition for height at the end of the canal
    E(end,:) = 4/3*E(end-1,:) - 1/3*E(end-2,:);

    % Tides at the inlet of the canal
    E(1,:) = 1*sin(t*2*pi/(3600*12));
%     if t < 1
%         E(1,:) = 0.1;
%     end

end