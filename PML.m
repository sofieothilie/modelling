classdef PML < handle
    % PML The handle class for the pml simulation.
    %     PML is the handle class for the 2D (and 1D) wave simulation with
    %     a reflectionless discrete pml.
    %
    %     We use the notation that if the domain grid is of size Nx*Ny,
    %     then a 1D array f of shape [Nx*Ny,1] is regarded as a function 
    %        f : {1,...,Nx*Ny} -> R (real number)
    %     so that the standard matlab call f(i) agrees with function 
    %     evaluation.  Similarly a 2D array F of shape [Nx,Ny] is a
    %     function
    %        f : {1,...,Nx} x {1,...,Ny} -> R.
    %     For convenience, let
    %        INDx := {1,...,Nx}, INDy := {1,...,Ny}, IND := {1,...,Nx*Ny}.
    %
    % PML Properties:
    %     (Domain parameters)
    %     xmin, ymin - coordinate of the bottom left corner of the domain.
    %     Lx, Ly - domain width and height (pml included).
    %     Nx, Ny - grid resolution.
    %
    %     (Pml parameters)
    %     Mx, My - number of grids for the pml.  They are used only for
    %          initializing the values of sigmax and sigmay.
    %     sigmax, sigmay - values of the pml damping coefficients. 
    %          sigmax : IND -> R,    sigmay : IND -> R.
    %          Note that sigma's are the rescaled coefficients; i.e.
    %          sigmax = [sigmax in the paper]*dx.  By default sigma's are
    %          set to be 2 in the layer and 0 in the physical domain.  User
    %          may modify the values of sigmax, sigmay.
    %
    %     (Domain intrinsic values and point attributes)
    %     dx, dy - grid size; i.e. Lx/Nx and Ly/Ny respectively.
    %     dt - time step; set to min([dx,dy])/2 by default.  User may
    %          modify the value of dt.
    %     X - physical x coordinate.  X : IND -> R.
    %     Y - physical y coordinate.  Y : IND -> R.
    %     Ix - Ix : IND -> INDx.  (for reading-off 2D array index).
    %     Iy - Iy : IND -> INDy.  (for reading-off 2D array index).
    %     I -  I  : IND -> IND.  Identity map (for reading-off linear index).
    %     IL - IL : IND -> IND.  Left-shift map.
    %     IR - IR : IND -> IND.  Right-shift map.
    %     IU - IU : IND -> IND.  Up-shift map.
    %     ID - ID : IND -> IND.  Down-shift map.
    %
    %     (Field variables)
    %     U - U : IND -> R.  The primary field for the scalar wave eq.
    %     V - V : IND -> R.  V = dU/dt.
    %     PHIx, PHIy, PSIx, PSIy - Auxilary variables.  They are all
    %          functions of the type IND -> R, but meaningful only in 
    %          the pml.
    %
    % PML Methods:
    %     PML - class constructor.
    %     build - helper function for the constructor.
    %     odefcn - implementation of the semi-discrete PDE called by rk4.
    %     RK4step - updates field variables by one step RK4 solve.
    %     createReferenceProblem - creates another PML handle with a larger
    %          domain for generating reference solutions.
    %     ind2sub - ind2sub function with Nx Ny shape.
    %     vec2grid - converts an (IND -> R) function into an (INDx x INDy
    %          -> R) 2D array.  Useful for plotting solutions.
    %     vec2gridInterior - converts an (IND -> R) function into a 2D 
    %          array excluding pml.  Useful for plotting solutions.
    
    properties
        %%% Domain info
        
        xmin, ymin  % coordinate of the bottom left corner (real number).
        Lx,Ly       % domain physical size (real number).
        Nx,Ny,      % grid resolution (integer).
        
        %%% pml info
        
        Mx,My,      % width of PML in number of grids (integer).
        sigmax      % scaled damping coefficients (IND -> R).
        sigmay      % scaled damping coefficients (IND -> R).
        
        %%% other domain info
        
        dx,dy       % grid size (real number); i.e. Lx/Nx.
        dt          % time step size (real number).
        X,Y         % grid coordinate (IND -> R).
        Ix          % linear to sub index (IND -> INDx).
        Iy          % linear to sub index (IND -> INDy).
        I           % identity map (IND -> IND).
        IL          % left shifting map (IND -> IND).
        IR          % right shifting map (IND -> IND).
        IU          % up shifting map (IND -> IND).
        ID          % down shifting map (IND -> IND).
        
        % variables
        U           % primary field for wave equation (IND -> R).
        V           % dU/dt (IND -> R).
        PHIx        % auxilary field (IND -> R).
        PSIx        % auxilary field (IND -> R).
        PHIy        % auxilary field (IND -> R).
        PSIy        % auxilary field (IND -> R).
    end
    methods
        %% CONSTRUCTOR
        function pml = PML(varargin)
        % PML class constructor.
        %
        % Syntax:
        %     pml = PML(); (empty object)
        %     pml = PML(xmin,ymin,Lx,Ly,Nx,Ny,Mx,My);
        %
        
            switch nargin
                case 0
                    return
                case 8
                    pml.xmin = varargin{1};
                    pml.ymin = varargin{2};
                    pml.Lx   = varargin{3};
                    pml.Ly   = varargin{4};
                    pml.Nx   = varargin{5};
                    pml.Ny   = varargin{6};
                    pml.Mx   = varargin{7};
                    pml.My   = varargin{8};
                    pml.build();
                otherwise
                    error('PML constructor inputs: xmin,ymin,Lx,Ly,Nx,Ny,Mx,My');
            end
        end
        
        %% BUILD/INITIALIZER
        function build(pml)
        % build - helper function for the constructor.
        %     Given xmin, ymin, Lx, Ly, Nx, Ny, Mx, My, compute/initialize
        %     all other class properties.
            pml.dx = pml.Lx/pml.Nx;
            pml.dy = pml.Ly/pml.Ny;
            pml.dt = min([pml.dx,pml.dy])/2;
            [pml.Ix,pml.Iy] = ndgrid(1:pml.Nx,1:pml.Ny);
            pml.Ix = pml.Ix(:);
            pml.Iy = pml.Iy(:);
            pml.X = pml.xmin + (pml.Ix-1)*pml.dx;
            pml.Y = pml.ymin + (pml.Iy-1)*pml.dy;
            pml.I  = pml.sub2ind(pml.Ix,pml.Iy);
            pml.IL = pml.sub2ind(pml.Ix-1,pml.Iy);
            pml.IR = pml.sub2ind(pml.Ix+1,pml.Iy);
            pml.ID = pml.sub2ind(pml.Ix,pml.Iy-1);
            pml.IU = pml.sub2ind(pml.Ix,pml.Iy+1);
            
            pml.sigmax = zeros(size(pml.I));
            pml.sigmay = zeros(size(pml.I));
            pml.sigmax(pml.Ix > pml.Nx-pml.Mx) = 2;
            pml.sigmay(pml.Iy > pml.Ny-pml.My) = 2;
            
            pml.U = zeros(size(pml.I));
            pml.V = pml.U;
            pml.PHIx = pml.U;
            pml.PHIy = pml.U;
            pml.PSIx = pml.U;
            pml.PSIy = pml.U;
        end
        
        %% ODE FUNCTION
        function yout = odefcn(pml,yin)
        % odefcn - implementation of the semi-discrete PDE called by rk4.
        %     Regarding the semi-discrete PDE as an ODE of the form
        %        dy/dt = F(y),
        %     this function is the implementation of F.  Here y is the
        %     variables packed like y=[U;V;PHIx;PSIx;PHIy;PSIy].
        %     
        % Syntax:
        %     yout = pml.odefcn(yin);
        %
        % See also RK4STEP
        
            % unpack input
            NN = pml.Nx * pml.Ny;
            u    = yin(     1 :   NN);
            v    = yin(  NN+1 : 2*NN);
            phix = yin(2*NN+1 : 3*NN);
            psix = yin(3*NN+1 : 4*NN);
            phiy = yin(4*NN+1 : 5*NN);
            psiy = yin(5*NN+1 : 6*NN);
            
            % PDE expression (Eq.(3) of the paper)
            du = v;
            dv =   ( u(pml.IL) - 2*u + u(pml.IR) )/(pml.dx^2) ...
                 + ( u(pml.ID) - 2*u + u(pml.IU) )/(pml.dy^2) ...
                 + ( pml.sigmax(pml.I ).*psix(pml.IR) ...
                   - pml.sigmax(pml.IL).*phix(pml.IL) )/(pml.dx^2) ...
                 + ( pml.sigmay(pml.I ).*psiy(pml.IU) ...
                   - pml.sigmay(pml.ID).*phiy(pml.ID) )/(pml.dy^2);
            dphix = -( pml.sigmax(pml.IL).*phix(pml.IL) ...
                     + pml.sigmax(pml.I ).*phix(pml.I ) )/(2*pml.dx) ...
                    -( u(pml.IR)-u(pml.IL) )/(2*pml.dx);
            dpsix = -( pml.sigmax(pml.IL).*psix(pml.I ) ...
                     + pml.sigmax(pml.I ).*psix(pml.IR) )/(2*pml.dx) ...
                    -( u(pml.IR)-u(pml.IL) )/(2*pml.dx);
            dphiy = -( pml.sigmay(pml.ID).*phiy(pml.ID) ...
                     + pml.sigmay(pml.I ).*phiy(pml.I ) )/(2*pml.dy) ...
                    -(u(pml.IU)-u(pml.ID) )/(2*pml.dy);
            dpsiy = -( pml.sigmay(pml.ID).*psiy(pml.I )...
                     + pml.sigmay(pml.I ).*psiy(pml.IU) )/(2*pml.dy) ...
                    -( u(pml.IU)-u(pml.ID) )/(2*pml.dy);
                
            % pack output
            yout = [du;dv;dphix;dpsix;dphiy;dpsiy];
        end
        %% RK4 SOLVER
        function RK4step(pml)
        % RK4step - updates field variables by one step RK4 solve.
        %     Regarding the semi-discrete PDE as an ODE of the form
        %        dy/dt = odefcn(y),
        %     RK4step solves the ODE for one time step and updates the
        %     values of the field variables.
        %     (y is the variables packed like y=[U;V;PHIx;PSIx;PHIy;PSIy].)
        %     
        % Syntax:
        %     pml.RK4step();
        %
        % See also ODEFCN
        
            % pack variables
            y = [pml.U;pml.V;pml.PHIx;pml.PSIx;pml.PHIy;pml.PSIy];
            
            % standard RK4
            k1 = pml.odefcn(y);
            k2 = pml.odefcn(y + pml.dt/2*k1);
            k3 = pml.odefcn(y + pml.dt/2*k2);
            k4 = pml.odefcn(y + pml.dt*k3);
            y = y+pml.dt/6*( k1 + 2*k2 + 2*k3 + k4 );
            
            % unpack variables
            NN = pml.Nx * pml.Ny;
            pml.U    = y(     1 :   NN);
            pml.V    = y(  NN+1 : 2*NN);
            pml.PHIx = y(2*NN+1 : 3*NN);
            pml.PSIx = y(3*NN+1 : 4*NN);
            pml.PHIy = y(4*NN+1 : 5*NN);
            pml.PSIy = y(5*NN+1 : 6*NN);
        end
        
        %% CREATE REFERENCE PROBLEM
        function [ref,Inj] = createReferenceProblem(pml,multx,multy)
        % createReferenceProblem - creates another PML handle obj with larger domain.
        %     This method creates another PML handle object with larger 
        %     domain but with zero pml.  This is for generating reference 
        %     solutions.  It returns not only a PML object but also an
        %     index map identifying the common domains.
        %
        % Syntax:
        %     [ref,Inj] = pml.createReferenceProblem(multx,multy);
        %
        % Description:
        %     The output ref is a PML object with Nx (resp. Ny) multx times
        %     (resp. multy times) larger than the Nx (resp. Ny) of the
        %     input PML object pml.  The output array Inj is the inclusion
        %     map
        %         Inj : [IND of pml] -> [IND of ref].
        %
        %     To extract only the common physical domain, use the pullback
        %     by pml.vec2gridInterior.
        %
        
        
            % ref is another PML domain multx x multy larger
            % Inj is the index as the inclusion map pml->ref
            ref = PML(pml.xmin,pml.ymin,...
                pml.Lx*multx,pml.Ly*multy,...
                pml.Nx*multx,pml.Ny*multy,0,0);
            ref.dt = pml.dt;
            Inj = find( ref.Ix<=pml.Nx & ref.Iy<=pml.Ny );
        end
        
        %% INDEXING HELPERS
        function ind = sub2ind(pml,subx,suby)
        % sub2ind - sub2ind function using the Nx Ny shape.
        %
        % Syntax:
        %     ind = pml.sub2ind(subx,suby);
        %
        
            periodify = @(ind,N) mod(ind-1,N)+1;
            ind = sub2ind([pml.Nx,pml.Ny],...
                    periodify(subx,pml.Nx),periodify(suby,pml.Ny));
        end
        function vv = vec2grid(pml,v)
        % vec2grid - converts a linear array into a 2D array.
        %
        % Syntax:
        %     vv = pml.vec2grid(v);
        %
        % Description:
        %     It reshapes an (IND -> R) function v into an (INDx x INDy
        %     -> R) 2D array vv.
        %
        % Examples:
        %     surf( pml.vec2grid(pml.X),...
        %           pml.vec2grid(pml.Y),...
        %           pml.vec2grid(pml.U));
        %
        % See also VEC2GRIDINTERIOR
        
            vv = reshape(v,pml.Nx,pml.Ny);
        end
        function vv = vec2gridInterior(pml,v)
        % vec2gridInterior - converts a linear array ito a 2D array.
        % Syntax:
        %     vv = pml.vec2gridInterior(v);
        %
        % Description:
        %     It reshapes an (IND -> R) function v into a 2D array of shape
        %     [Nx-Mx, Ny-My].  That is, it excludes the pml.
        %
        % Examples:
        %     surf( pml.vec2gridInterior(pml.X),...
        %           pml.vec2gridInterior(pml.Y),...
        %           pml.vec2gridIntrior(pml.U));
        %
        % See also VEC2GRID
        
            vv = reshape(v(pml.Ix<=pml.Nx-pml.Mx & pml.Iy<=pml.Ny-pml.My),...
                pml.Nx-pml.Mx,pml.Ny-pml.My);
        end
        
        %% VPA TOOL
        function makevpa(pml,d)
        % makevpa - make all variables have variable precision (symbolic tool) 
            if nargin==2
                digits(d);
            end
            pml.U = vpa(pml.U);
            pml.V = vpa(pml.V);
            pml.PHIx = vpa(pml.PHIx);
            pml.PHIy = vpa(pml.PHIy);
            pml.PSIx = vpa(pml.PSIx);
            pml.PSIy = vpa(pml.PSIy);
        end
    end
end