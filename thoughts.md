## formula:

The variables `u1p` and `u1pp` refer to the previous step of the simulation, and the previous previous step of the simulation, respectively. So if `u1` is at `t=10` then `u1p`  is at `t=9` and `u1pp` is at `t=8`. Same with `u2` and `u3`. 

```
Ux_nxt[x, y, z] = (dt**2*((lambda[x, y, z] + mu[x, y, z])*(Uy[x - 1, y - 1, z] - Uy[x - 1, y + 1, z] - Uy[x + 1, y - 1, z] + Uy[x + 1, y + 1, z]) + (lambda[x, y, z] + mu[x, y, z])*(Uz[x - 1, y, z - 1] - Uz[x - 1, y, z + 1] - Uz[x + 1, y, z - 1] + Uz[x + 1, y, z + 1]) + 4*(lambda[x, y, z] + 2*mu[x, y, z])*(-2*Ux[x, y, z] + Ux[x - 1, y, z] + Ux[x + 1, y, z]) + 4*(-2*Ux[x, y, z] + Ux[x, y, z - 1] + Ux[x, y, z + 1])*mu[x, y, z] + 4*(-2*Ux[x, y, z] + Ux[x, y - 1, z] + Ux[x, y + 1, z])*mu[x, y, z])/4 + (2*Ux[x, y, z] - Ux_prv[x, y, z])*rho[x, y, z])/rho[x, y, z];

Uy_nxt[x, y, z] = (dt**2*((lambda[x, y, z] + mu[x, y, z])*(Ux[x - 1, y - 1, z] - Ux[x - 1, y + 1, z] - Ux[x + 1, y - 1, z] + Ux[x + 1, y + 1, z]) + (lambda[x, y, z] + mu[x, y, z])*(Uz[x, y - 1, z - 1] - Uz[x, y - 1, z + 1] - Uz[x, y + 1, z - 1] + Uz[x, y + 1, z + 1]) + 4*(lambda[x, y, z] + 2*mu[x, y, z])*(-2*Uy[x, y, z] + Uy[x, y - 1, z] + Uy[x, y + 1, z]) + 4*(-2*Uy[x, y, z] + Uy[x, y, z - 1] + Uy[x, y, z + 1])*mu[x, y, z] + 4*(-2*Uy[x, y, z] + Uy[x - 1, y, z] + Uy[x + 1, y, z])*mu[x, y, z])/4 + (2*Uy[x, y, z] - Uy_prv[x, y, z])*rho[x, y, z])/rho[x, y, z];

Uz_nxt[x, y, z] = (dt**2*((lambda[x, y, z] + mu[x, y, z])*(Ux[x - 1, y, z - 1] - Ux[x - 1, y, z + 1] - Ux[x + 1, y, z - 1] + Ux[x + 1, y, z + 1]) + (lambda[x, y, z] + mu[x, y, z])*(Uy[x, y - 1, z - 1] - Uy[x, y - 1, z + 1] - Uy[x, y + 1, z - 1] + Uy[x, y + 1, z + 1]) + 4*(lambda[x, y, z] + 2*mu[x, y, z])*(-2*Uz[x, y, z] + Uz[x, y, z - 1] + Uz[x, y, z + 1]) + 4*(-2*Uz[x, y, z] + Uz[x, y - 1, z] + Uz[x, y + 1, z])*mu[x, y, z] + 4*(-2*Uz[x, y, z] + Uz[x - 1, y, z] + Uz[x + 1, y, z])*mu[x, y, z])/4 + (2*Uz[x, y, z] - Uz_prv[x, y, z])*rho[x, y, z])/rho[x, y, z];
```
## basic concept

this formula computes the next step from 3 previous steps.

- What is my initial situation? -> wave started from sender
- should I compute the formula for the whole space at the same time, then triple for loop over whole model ?
- until when do we compute ? 
- do I need 2 steps to start the simulation correctly ? (uprev and u)
- wait why does the formula use the current value of other dims ? I don't think I can use the current time neighbors to determine myself

- try random value in middle to test

- export data: export all frames (or 1 slice and visualize it)

- 


## actual use of the formula

I don't want to store a whole matrix for lambda, mu and rho, but I can make functions that compute their value at this position, looking at the model.

Then I keep in memory u1, u2, u3, and their p and pp version. => 9 full matrix, which would be quite big (model is 300x400x1200 or smth)

## implementation

what spatial scale do we use ? I guess indices for the whole space => so dx = 1, but we can change this later.

ok so the model seems to be stored as a 2d matrix of the height of the thing. then I only need to know 

Then i can just check on the x-y dimension if my current z (depth) coordinate is in the model or not

So what do I need ? I need the height of the plastic model. and to determine what I have where




### code structure

- init memory
- time step
- output data (what do we want to see ?)
- destroy memory

->basically just copy the 2d example, in 3d, with updated formula

### optimization

/

### upgrades

- add air over water,and take into account in simulation since wave can go out and come back

### debugging

I currently get a wave, looks correct for plastic, but weird for water. It seems to follow a specific dimension more, but I guess that's normal (yeah its probably normal),
Here's my explanation, I apply a sine wave in all directions, but here I observe only the X axis, so its normal that I see more along X axis than others, it cant  spread as easily in other dimensions.


- why does the center point moves at 2 places when res (10,40,100)

### previous formulas

    // //1st
    // P_nxt(i, j, k) = (dt*dt*(dx*dx*dy*dy*((K(i, j, k - 1) - K(i, j, k + 1))*(K(i, j, k - 1) - K(i, j, k + 1))*P(i, j, k) + 2*(K(i, j, k - 1) - K(i, j, k + 1))*(P(i, j, k - 1) - P(i, j, k + 1))*K(i, j, k) + 4*(-2*K(i, j, k) + K(i, j, k - 1) + K(i, j, k + 1))*K(i, j, k)*P(i, j, k) + 2*(-2*P(i, j, k) + P(i, j, k - 1) + P(i, j, k + 1))*K(i, j, k)*K(i, j, k))
    //  + dx*dx*dz*dz*((K(i, j - 1, k) - K(i, j + 1, k))*(K(i, j - 1, k) - K(i, j + 1, k))*P(i, j, k) + 2*(K(i, j - 1, k) - K(i, j + 1, k))*(P(i, j - 1, k) - P(i, j + 1, k))*K(i, j, k) + 4*(-2*K(i, j, k) + K(i, j - 1, k) + K(i, j + 1, k))*K(i, j, k)*P(i, j, k) + 2*(-2*P(i, j, k) + P(i, j - 1, k) + P(i, j + 1, k))*K(i, j, k)*K(i, j, k)) 
    //  + dy*dy*dz*dz*((K(i - 1, j, k) - K(i + 1, j, k))*(K(i - 1, j, k) - K(i + 1, j, k))*P(i, j, k) + 2*(K(i - 1, j, k) - K(i + 1, j, k))*(P(i - 1, j, k) - P(i + 1, j, k))*K(i, j, k) + 4*(-2*K(i, j, k) + K(i - 1, j, k) + K(i + 1, j, k))*K(i, j, k)*P(i, j, k) + 2*(-2*P(i, j, k) + P(i - 1, j, k) + P(i + 1, j, k))*K(i, j, k)*K(i, j, k))) 
    //  + 2*dx*dx*dy*dy*dz*dz*(2*P(i, j, k) - P_prv(i, j, k)))/(2*dx*dx*dy*dy*dz*dz);

    //3nd: larger stencil
    // P_nxt(i, j, k) = (dt*dt*(dx*dx*dy*dy*(((K(i, j, k - 3) - 9*K(i, j, k - 2) + 45*K(i, j, k - 1) - 45*K(i, j, k + 1) + 9*K(i, j, k + 2) - K(i, j, k + 3))*P(i, j, k) + (P(i, j, k - 3) - 9*P(i, j, k - 2) + 45*P(i, j, k - 1) - 45*P(i, j, k + 1) + 9*P(i, j, k + 2) - P(i, j, k + 3))*K(i, j, k))*(K(i, j, k - 3) - 9*K(i, j, k - 2) + 45*K(i, j, k - 1) - 45*K(i, j, k + 1) + 9*K(i, j, k + 2) - K(i, j, k + 3)) + 2*((K(i, j, k - 3) - 9*K(i, j, k - 2) + 45*K(i, j, k - 1) - 45*K(i, j, k + 1) + 9*K(i, j, k + 2) - K(i, j, k + 3))*(P(i, j, k - 3) - 9*P(i, j, k - 2) + 45*P(i, j, k - 1) - 45*P(i, j, k + 1) + 9*P(i, j, k + 2) - P(i, j, k + 3)) + 10*(-490*K(i, j, k) + 2*K(i, j, k - 3) - 27*K(i, j, k - 2) + 270*K(i, j, k - 1) + 270*K(i, j, k + 1) - 27*K(i, j, k + 2) + 2*K(i, j, k + 3))*P(i, j, k) + 10*(-490*P(i, j, k) + 2*P(i, j, k - 3) - 27*P(i, j, k - 2) + 270*P(i, j, k - 1) + 270*P(i, j, k + 1) - 27*P(i, j, k + 2) + 2*P(i, j, k + 3))*K(i, j, k))*K(i, j, k)) + dx*dx*dz*dz*(((K(i, j - 3, k) - 9*K(i, j - 2, k) + 45*K(i, j - 1, k) - 45*K(i, j + 1, k) + 9*K(i, j + 2, k) - K(i, j + 3, k))*P(i, j, k) + (P(i, j - 3, k) - 9*P(i, j - 2, k) + 45*P(i, j - 1, k) - 45*P(i, j + 1, k) + 9*P(i, j + 2, k) - P(i, j + 3, k))*K(i, j, k))*(K(i, j - 3, k) - 9*K(i, j - 2, k) + 45*K(i, j - 1, k) - 45*K(i, j + 1, k) + 9*K(i, j + 2, k) - K(i, j + 3, k)) + 2*((K(i, j - 3, k) - 9*K(i, j - 2, k) + 45*K(i, j - 1, k) - 45*K(i, j + 1, k) + 9*K(i, j + 2, k) - K(i, j + 3, k))*(P(i, j - 3, k) - 9*P(i, j - 2, k) + 45*P(i, j - 1, k) - 45*P(i, j + 1, k) + 9*P(i, j + 2, k) - P(i, j + 3, k)) + 10*(-490*K(i, j, k) + 2*K(i, j - 3, k) - 27*K(i, j - 2, k) + 270*K(i, j - 1, k) + 270*K(i, j + 1, k) - 27*K(i, j + 2, k) + 2*K(i, j + 3, k))*P(i, j, k) + 10*(-490*P(i, j, k) + 2*P(i, j - 3, k) - 27*P(i, j - 2, k) + 270*P(i, j - 1, k) + 270*P(i, j + 1, k) - 27*P(i, j + 2, k) + 2*P(i, j + 3, k))*K(i, j, k))*K(i, j, k)) + dy*dy*dz*dz*(((K(i - 3, j, k) - 9*K(i - 2, j, k) + 45*K(i - 1, j, k) - 45*K(i + 1, j, k) + 9*K(i + 2, j, k) - K(i + 3, j, k))*P(i, j, k) + (P(i - 3, j, k) - 9*P(i - 2, j, k) + 45*P(i - 1, j, k) - 45*P(i + 1, j, k) + 9*P(i + 2, j, k) - P(i + 3, j, k))*K(i, j, k))*(K(i - 3, j, k) - 9*K(i - 2, j, k) + 45*K(i - 1, j, k) - 45*K(i + 1, j, k) + 9*K(i + 2, j, k) - K(i + 3, j, k)) + 2*((K(i - 3, j, k) - 9*K(i - 2, j, k) + 45*K(i - 1, j, k) - 45*K(i + 1, j, k) + 9*K(i + 2, j, k) - K(i + 3, j, k))*(P(i - 3, j, k) - 9*P(i - 2, j, k) + 45*P(i - 1, j, k) - 45*P(i + 1, j, k) + 9*P(i + 2, j, k) - P(i + 3, j, k)) + 10*(-490*K(i, j, k) + 2*K(i - 3, j, k) - 27*K(i - 2, j, k) + 270*K(i - 1, j, k) + 270*K(i + 1, j, k) - 27*K(i + 2, j, k) + 2*K(i + 3, j, k))*P(i, j, k) + 10*(-490*P(i, j, k) + 2*P(i - 3, j, k) - 27*P(i - 2, j, k) + 270*P(i - 1, j, k) + 270*P(i + 1, j, k) - 27*P(i + 2, j, k) + 2*P(i + 3, j, k))*K(i, j, k))*K(i, j, k))) + 3600*dx*dx*dy*dy*dz*dz*(2*P(i, j, k) - P_prv(i, j, k)))/(3600*dx*dx*dy*dy*dz*dz);
    