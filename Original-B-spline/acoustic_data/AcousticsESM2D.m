clear all;
close all;
clc;
%  note1: adjust the position of eq sources when divergent
%  note2: the position of sources must match the number of control points:
%         the result of ng = 400, dn = 1, dr = 0.01 is better than
%                       ng = 400, dn = 1, dr = 0.005
%  the result should be correct while the sources are loacted in 
%             appropriate regions
% constrained ESMon the continuty of 
format long
proname='circle';
smeshflag = 0;
sK = 1;   % selected wave number
RR = 40;
spos = [0,-5,0]; %for 2D cases: [*,*,0]; outside the body
soundfname = ['result/ESM_mono_',proname,'_smesh',num2str(smeshflag),'_k=',...
    num2str(sK),'_','x_',num2str(spos(1)),...
    '_y_',num2str(spos(2)),'.dat'];
%%
gmesh = load([proname,'/gmesh.dat']); 
%gmesh = load('circle/gmesh.dat');
ng = length(gmesh(:,1));
%% 
smesh = dlmread([proname,'/smesh.dat']);
if (smeshflag == 1)
smesh = dlmread([proname,'/smesh1.dat']);
end
%smesh = load('circle/smesh.dat'); 
ns = length(smesh(:,1));
%% 
delta = 0.1; % delta need to be adjusted with sK if used

sA = 1.0;   % the source amplitude
uniti = (-1)^0.5;
lamda = 10^(-5); % the parameter of tikhonuv
lamda1 = 10^(-6); % the parameter for svd
%% 
for i = 1:ng    
   for j= 1:ns
       r = gmesh(i,1:3)-smesh(j,:);
       %dis = norm(r);
       dis = (r(1)^2+r(2)^2+r(3)^2)^0.5;
       prpn = r(1)*gmesh(i,4)+r(2)*gmesh(i,5)+...
                r(3)*gmesh(i,6);
       A(i,j) = prpn*sK/dis*uniti/8.0*(exp(-pi*uniti)-1)*besselh(1,1,sK*dis); 
       %A(i,j) = prpn*sK/dis/4.0/uniti*(besselh(0,1,sK*dis+delta)-besselh(0,1,sK*dis-delta))/2/delta;
   end   
   r1 = gmesh(i,1:3)-spos;
   %dis = norm(r1);
   dis = (r1(1)^2+r1(2)^2+r1(3)^2)^0.5;
   prpn = r1(1)*gmesh(i,4)+r1(2)*gmesh(i,5)+...
                r1(3)*gmesh(i,6);
   b(i) = -sA*prpn/dis*sK*uniti/8.0*(exp(-pi*uniti)-1)*besselh(1,1,sK*dis);
   
   %b(i) = -sA*prpn/dis*sK/uniti/4.0*(besselh(0,1,sK*dis+delta)-besselh(0,1,sK*dis-delta))/2/delta;
   if (mod(i,1000)==0)
       fprintf('the number i is: %d', i);
       fprintf('\n');
   end
end
tic

%% SVD
[s1,v1,d1] = svd(A);
diagv1 = diag(v1);
b1 = inv(s1)*b.';
%lamda2 = max((diagv1))*min((diagv1));
%lamda2 = 10^(-6);
lamda2 = 0;
for j = 1:ns
    b2(j) = b1(j)/(diagv1(j)+lamda2/diagv1(j));
end
% %x = gmres(d1',b2,restart,tol,maxit);
xx = d1*b2.';
bx = A*xx;
% 
error1 = norm(bx.'-b)/norm(b)

%return
%% exam one point
amp = xx;
dl = 1; nx = 100; ny= 100;
x1 = -dl*nx/2; y1 = -dl*ny/2;
numNodes = (nx+1)*(ny+1);
numEle = nx*ny;
    des = [500,-800,0];
    C(1,1) = 0;
    C(1,2) = 0;
    for i = 1:ns
        r = des - smesh(i,:);
        %dis = norm(r);
        dis = (r(1)^2+r(2)^2+r(3)^2)^0.5;
        C(1,1) = C(1,1)+amp(i)*besselh(0,1,sK*dis)/4/uniti;
        C(1,2) = C(1,2)+amp(i)*besselh(0,1,sK*dis)/4/uniti;
    end
    C1 = abs(C(1,1))
    C2 = angle(C(1,2))
    %return
    %rpos = des - gcen;
    %if (norm(rpos)<=rat)
    %C(ii+(jj-1)*(nx+1),1:2) =  C(ii+(jj-1)*(nx+1),1:2)- C(ii+(jj-1)*(nx+1),1:2);
    %end
%return

%% exam points on a line
amp = xx;
np = 180;
    D(1,1) = 0;
    D(1,2) = 0;
    for j = 1:np
        cta(j) = 2*pi/np*j;
        des = [RR*cos(cta(j)),RR*sin(cta(j)),0];
        Linep(j,1)=0;
        Linep(j,2)=0;
        CC = 0;
        for i = 1:ns        
            r = des - smesh(i,:);
            %dis = norm(r);
            dis = (r(1)^2+r(2)^2+r(3)^2)^0.5;
            CC = CC+amp(i)*besselh(0,1,sK*dis)/4/uniti;
        end
        Linep(j,1) = abs(CC);
        Linep(j,2) = angle(CC);
    end
    outLine = [cta;Linep']';
    dlmwrite(['result/','ESM_Line_',proname,'_smesh',num2str(smeshflag),'_k=',...
    num2str(sK),'_','x_',num2str(spos(1)),...
    '_y_',num2str(spos(2)),'.dat'],outLine);
    plot(Linep(:,1))
    
    %return
    %rpos = des - gcen;
    %if (norm(rpos)<=rat)
    %C(ii+(jj-1)*(nx+1),1:2) =  C(ii+(jj-1)*(nx+1),1:2)- C(ii+(jj-1)*(nx+1),1:2);
    %end
return

%% output 2D data for learning
x0 = 100; y0 = 100;
nx = 10; ny = 10;
Lx = 6; Ly = 6;
dx = Lx/nx; dy = Ly/ny; 
C_train = zeros(nx,ny,2);
nx1 = 50; ny1 = 50;
for ii = 1:nx
    for jj = 1:ny       
             CC = 0;
             des = [x0-Lx/2+dx*(ii-1), ...
                    y0-Ly/2+dy*(jj-1), ...
                    0];
             for i = 1:ng                    
                r = des - smesh(i,:);
                %dis = norm(r);
                dis = (r(1)^2+r(2)^2+r(3)^2)^0.5;
                CC = CC+amp(i)*besselh(0,1,sK*dis)/4/uniti;  
             end
            C_train(ii,jj,1) = des(1);
            C_train(ii,jj,2) = des(2);
            C_train(ii,jj,3) = abs(CC);
            C_train(ii,jj,4) = angle(CC);
    end
    fprintf('ii = %d \n', ii)
end
save('C_train_2D.mat', 'C_train');

%% output 2D data for plotting
x0 = 100; y0 = 100;
nx = 100; ny = 100;
Lx = 6.1; Ly = 6.1;
dx = Lx/nx; dy = Ly/ny; 
C_train = zeros(nx,ny,2);
nx1 = 50; ny1 = 50;
for ii = 1:nx
    for jj = 1:ny       
             CC = 0;
             des = [x0-Lx/2+dx*(ii-1), ...
                    y0-Ly/2+dy*(jj-1), ...
                    0];
             for i = 1:ng                    
                r = des - smesh(i,:);
                %dis = norm(r);
                dis = (r(1)^2+r(2)^2+r(3)^2)^0.5;
                CC = CC+amp(i)*besselh(0,1,sK*dis)/4/uniti;  
             end
            C_train_plot(ii,jj,1) = des(1);
            C_train_plot(ii,jj,2) = des(2);
            C_train_plot(ii,jj,3) = abs(CC);
            C_train_plot(ii,jj,4) = angle(CC);
    end
    fprintf('ii = %d \n', ii)
end
save('C_train_2D_plot.mat', 'C_train_plot');

%% output 2D data for plotting 1
x0 = 0; y0 = 0;
nx = 200; ny = 200;
Lx = 24; Ly = 24;
dx = Lx/nx; dy = Ly/ny; 
C_train = zeros(nx,ny,2);
nx1 = 50; ny1 = 50;
for ii = 1:nx
    for jj = 1:ny       
             CC = 0;
             des = [x0-Lx/2+dx*(ii-1), ...
                    y0-Ly/2+dy*(jj-1), ...
                    0];
             for i = 1:ng                    
                r = des - smesh(i,:);
                %dis = norm(r);
                dis = (r(1)^2+r(2)^2+r(3)^2)^0.5;
                CC = CC+amp(i)*besselh(0,1,sK*dis)/4/uniti;  
             end
            C_train_plot(ii,jj,1) = des(1);
            C_train_plot(ii,jj,2) = des(2);
            C_train_plot(ii,jj,3) = abs(CC);
            C_train_plot(ii,jj,4) = angle(CC);
    end
    fprintf('ii = %d \n', ii)
end
save('C_train_2D_plot_1.mat', 'C_train_plot');
%% plot results (validation)
%cta0 = 77/180*pi; r0 = 0.6;
%des = [r0*cos(cta0),r0*sin(cta0)];
%u0 = 1-cos(2*cta0)/r0^2
amp = xx;
dl = 0.2; nx = 100; ny= 100;
x1 = -dl*nx/2; y1 = -dl*ny/2;
numNodes = (nx+1)*(ny+1);
numEle = nx*ny;
for jj=1:ny+1
  for ii=1:nx+1
    des = [(ii-1)*dl+x1,(jj-1)*dl+y1,0];
    C(ii+(jj-1)*(nx+1),1) = 0;
    C(ii+(jj-1)*(nx+1),2) = 0;
    for i = 1:ns
        r = des - smesh(i,:);
        %dis = norm(r);
        dis = (r(1)^2+r(2)^2+r(3)^2)^0.5;
        C(ii+(jj-1)*(nx+1),1) = C(ii+(jj-1)*(nx+1),1)+amp(i)*besselh(0,1,sK*dis)/4/uniti;
        C(ii+(jj-1)*(nx+1),2) = C(ii+(jj-1)*(nx+1),2)+amp(i)*besselh(0,1,sK*dis)/4/uniti;
    end
    C(ii+(jj-1)*(nx+1),1) = log10(abs(C(ii+(jj-1)*(nx+1),1)));
    C(ii+(jj-1)*(nx+1),2) = angle(C(ii+(jj-1)*(nx+1),2));
  end
    if (mod(jj,100)==0)
       fprintf('the number jj is: %d', jj);
       fprintf('\n');
    end
    
end


%% plot 2D potential field
a=fopen(soundfname,'w'); 
fprintf(a,'TITLE   = "tecplot binary file"');
fprintf(a,'\n');
fprintf(a,'VARIABLES = "X"');
fprintf(a,'\n');
fprintf(a,'"Y"');
fprintf(a,'\n');
fprintf(a,'"amp"');
fprintf(a,'\n');
fprintf(a,'"angle"');
fprintf(a,'\n');
fprintf(a,'ZONE T="square zone"');
fprintf(a,'\n');
fprintf(a,'STRANDID=0, SOLUTIONTIME=0');
fprintf(a,'\n');
 fprintf(a,[' Nodes=', num2str(numNodes),',Elements=',...
        num2str(numEle),',ZONETYPE=FEQuadrilateral']);
fprintf(a,'\n');
fprintf(a,'DATAPACKING=POINT');
fprintf(a,'\n');
fprintf(a,' DT=(SINGLE SINGLE SINGLE SINGLE SINGLE )');

for jj=1:ny+1
    for ii=1:nx+1
         fprintf(a,'\n');
         if (proname=='Square')
            if (inoutsquare((ii-1)*dl+x1,(jj-1)*dl+y1,gmesh)==0)
                C(ii+(jj-1)*(nx+1),1:2)=0;
            end
         end
         fprintf(a,'%d,%d,%d,%d,%d',(ii-1)*dl+x1,(jj-1)*dl+y1,C(ii+(jj-1)*(nx+1),1:2)); 
    end
end

for jj=1:ny
    for ii=1:nx
         fprintf(a,'\n');
         fprintf(a,'%d,%d,%d,%d',ii+(jj-1)*(nx+1),ii+(jj-1)*(nx+1)+1,...
                    ii+(jj-1)*(nx+1)+1+nx+1,ii+(jj-1)*(nx+1)+(nx+1)); 
    end
end
fclose(a);


