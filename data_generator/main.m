% Numeric parameters

x0 = 0;
x1 = 10000;
nx = 128;

y0 = -250;
y1 = 250;
ny = 32;

tf = 24*3600;
dt = 1;

pars.g = 9.81;
pars.C = 0.2;

finiteDifferencesMethod = 'EX4';
timeSteppingMethod = 'RK4';
meshType = 'uniform';

saveSteps = 3600;
plotSteps = 3600;
saveImages = false;

%% Generate mesh

X = linspace(x0,x1,nx);
Y = linspace(y0,y1,ny);

pars.deltaX = (x1-x0)/nx;
pars.deltaY = (y1-y0)/ny;

T = 0:dt*saveSteps:tf;
nt = length(T);

%% Define depth
pars.H = 15 - 10*(X'/x1) - 2*abs(Y/y1);

%% Generate initial conditions

E0 = zeros(nx,ny);
U0 = zeros(nx,ny);
V0 = zeros(nx,ny);

%% Initialize solUtion
E = nan(nx,ny,nt);
U = nan(nx,ny,nt);
V = nan(nx,ny,nt);

E(:,:,1) = E0;
U(:,:,1) = U0;
V(:,:,1) = V0;

%% Generate matrices for finite differences
dx = finiteDifferencesFunc(X,finiteDifferencesMethod,1,false);
dy = finiteDifferencesFunc(Y,finiteDifferencesMethod,2,false);

filter = spatialFilterFunc(nx,ny,0.4,false,false);

%% Time stepping

Eold = E0;
Uold = U0;
Vold = V0;
nSave = 1;
nImage = 1;

surfStep = 1;
quiverStep = 1;
[Xquiver,Yquiver] = meshgrid(X(1:quiverStep:end),Y(1:quiverStep:end),0);
Wquiver = 0*U0(1:quiverStep:end,1:quiverStep:end)';

for i = 1:(nt-1)*saveSteps
    
    disp(i)
    
    [Enew, Unew, Vnew] = timestepper((i-1)*dt,Eold,Uold,Vold,timeSteppingMethod,dt,pars,dx,dy,filter);
    
    if mod(i,saveSteps)==0
        nSave = nSave + 1;
        
        E(:,:,nSave) = Enew;
        U(:,:,nSave) = Unew;
        V(:,:,nSave) = Vnew;
    end

    if mod(i,plotSteps)==0
        clf
        hold on
        colormap(squeeze(hsv2rgb(0.6*ones(256,1),linspace(1,0.7,256)',linspace(0.3,1,256)')))
        surf(X(1:surfStep:end),Y(1:surfStep:end),Enew(1:surfStep:end,1:surfStep:end)')
        quiver3(Xquiver,Yquiver,0.05+Enew(1:quiverStep:end,1:quiverStep:end)',Unew(1:quiverStep:end,1:quiverStep:end)',Vnew(1:quiverStep:end,1:quiverStep:end)',Wquiver,'w')
        view(-15,45)
        axis([x0 x1 y0 y1 -1.5 1.5]);

%         subplot(3,1,1)
%         plot(X,Enew(:,ny/2))
%         subplot(3,1,2)
%         plot(X,Unew(:,ny/2))
%         subplot(3,1,3)
%         plot(X,Vnew(:,ny/2))

%         subplot(3,1,1)
%         contourf(X,Y,Enew')
%         colorbar
%         subplot(3,1,2)
%         contourf(X,Y,Unew')
%         colorbar
%         subplot(3,1,3)
%         contourf(X,Y,Vnew')
%         colorbar

        drawnow
        if saveImages
            saveas(gcf,['images/' num2str(nImage,'%04d') '.png']);
        end
        nImage = nImage + 1;

        %pause
    end
    
    if any(isnan(Enew(:)))
        pause
    end

    Eold = Enew;
    Uold = Unew;
    Vold = Vnew;
end

save(['ref_x' num2str(nx) '_y' num2str(ny) '_' finiteDifferencesMethod],'T','X','Y','U','V','E')