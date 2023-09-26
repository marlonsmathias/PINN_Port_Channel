function [E1,U1,V1] = timestepper(t,E0,U0,V0,method,dt,pars,dx,dy,filter)

    switch method
        case 'Euler'
            [Et,Ut,Vt] = nseq(E0,U0,V0,pars,dx,dy);
            E1 = E0 + dt*Et;
            U1 = U0 + dt*Ut;
            V1 = V0 + dt*Vt;

            [E1,U1,V1] = boundaryConditions(t+dt,E1,U1,V1);

            E1 = filter(E1);
            U1 = filter(U1);
            V1 = filter(V1);
            [E1,U1,V1] = boundaryConditions(t+dt,E1,U1,V1);

        case 'RK4'
            [Et1,Ut1,Vt1] = nseq(E0,U0,V0,pars,dx,dy);
            E1 = E0+dt/2*Et1;
            U1 = U0+dt/2*Ut1;
            V1 = V0+dt/2*Vt1;
            [E1,U1,V1] = boundaryConditions(t+dt/2,E1,U1,V1);

            [Et2,Ut2,Vt2] = nseq(E1,U1,V1,pars,dx,dy);
            E1 = E0+dt/2*Et2;
            U1 = U0+dt/2*Ut2;
            V1 = V0+dt/2*Vt2;
            [E1,U1,V1] = boundaryConditions(t+dt/2,E1,U1,V1);

            [Et3,Ut3,Vt3] = nseq(E1,U1,V1,pars,dx,dy);
            E1 = E0+dt*Et3;
            U1 = U0+dt*Ut3;
            V1 = V0+dt*Vt3;
            [E1,U1,V1] = boundaryConditions(t+dt,E1,U1,V1);

            [Et4,Ut4,Vt4] = nseq(E1,U1,V1,pars,dx,dy);

            E1 = E0 + dt/6*(Et1 + 2*Et2 + 2*Et3 + Et4);
            U1 = U0 + dt/6*(Ut1 + 2*Ut2 + 2*Ut3 + Ut4);
            V1 = V0 + dt/6*(Vt1 + 2*Vt2 + 2*Vt3 + Vt4);
            [E1,U1,V1] = boundaryConditions(t+dt,E1,U1,V1);

            E1 = filter(E1);
            U1 = filter(U1);
            V1 = filter(V1);
            [E1,U1,V1] = boundaryConditions(t+dt,E1,U1,V1);

    end

end