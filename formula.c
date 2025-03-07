//just a file to manipulate the formula:


//these functions just look at the model and determine the constant values from the medium
double lambda_at(int x, int y, int z);
double mu_at(int x, int y, int z);
double rho_at(int x, int y, int z);

double dt;

void timestep(int x, int y, int z){

    double lambda = lambda_at(x, y, z);
    double mu = mu_at(x, y, z);
    double rho = rho_at(x, y, z);

    //u is displacement vector (dx)

    //dx,dy,dz == 1 for now

    u1[x, y, z] = (dt*dt*((lambda + mu)*(u2[x - 1, y - 1, z] - u2[x - 1, y + 1, z] - u2[x + 1, y - 1, z] + u2[x + 1, y + 1, z]) 
                + (lambda + mu)*(u3[x - 1, y, z - 1] - u3[x - 1, y, z + 1] - u3[x + 1, y, z - 1] + u3[x + 1, y, z + 1]) 
                + 4*(lambda + 2*mu)*(-2*u1[x, y, z] + u1[x - 1, y, z] + u1[x + 1, y, z]) + 4*(-2*u1[x, y, z] + u1[x, y, z - 1] + u1[x, y, z + 1])*mu 
                + 4*(-2*u1[x, y, z] + u1[x, y - 1, z] + u1[x, y + 1, z])*mu)/4 + (2*u1p[x, y, z] - u1pp[x, y, z])*rho)/rho;

    u2[x, y, z] = (dt*dt*((lambda + mu)*(u1[x - 1, y - 1, z] - u1[x - 1, y + 1, z] - u1[x + 1, y - 1, z] + u1[x + 1, y + 1, z])
                + (lambda + mu)*(u3[x, y - 1, z - 1] - u3[x, y - 1, z + 1] - u3[x, y + 1, z - 1] + u3[x, y + 1, z + 1])
                + 4*(lambda + 2*mu)*(-2*u2[x, y, z] + u2[x, y - 1, z] + u2[x, y + 1, z]) + 4*(-2*u2[x, y, z] + u2[x, y, z - 1] + u2[x, y, z + 1])*mu 
                + 4*(-2*u2[x, y, z] + u2[x - 1, y, z] + u2[x + 1, y, z])*mu)/4 + (2*u2p[x, y, z] - u2pp[x, y, z])*rho)/rho;

    u3[x, y, z] = (dt*dt*((lambda + mu)*(u1[x - 1, y, z - 1] - u1[x - 1, y, z + 1] - u1[x + 1, y, z - 1] + u1[x + 1, y, z + 1]) 
                + (lambda + mu)*(u2[x, y - 1, z - 1] - u2[x, y - 1, z + 1] - u2[x, y + 1, z - 1] + u2[x, y + 1, z + 1]) 
                + 4*(lambda + 2*mu)*(-2*u3[x, y, z] + u3[x, y, z - 1] + u3[x, y, z + 1]) + 4*(-2*u3[x, y, z] + u3[x, y - 1, z] + u3[x, y + 1, z])*mu 
                + 4*(-2*u3[x, y, z] + u3[x - 1, y, z] + u3[x + 1, y, z])*mu)/4 + (2*u3p[x, y, z] - u3pp[x, y, z])*rho)/rho;
}